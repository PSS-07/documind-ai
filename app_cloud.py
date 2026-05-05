import streamlit as st
from auth import authenticator
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import tempfile

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
}
button {
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,0.2);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOGIN
# -------------------------------
st.markdown("### 🔐 Login to DocuMind")

name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("❌ Invalid username or password")

elif auth_status is None:
    st.warning("⚠️ Please enter your credentials")

# -------------------------------
# MAIN APP (ONLY IF LOGGED IN)
# -------------------------------
elif auth_status:

    authenticator.logout("Logout", "sidebar")

    # -------------------------------
    # Header
    # -------------------------------
    st.markdown("""
    <h1 style='text-align: center;'>
    📘 <span style="color:#38bdf8;">DocuMind AI</span>
    </h1>
    <p style='text-align: center; color: #94a3b8; font-size:18px;'>
    Chat with your documents like never before
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # -------------------------------
    # Session State
    # -------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "db" not in st.session_state:
        st.session_state.db = None

    # -------------------------------
    # Sidebar
    # -------------------------------
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📂 Upload Documents")
        st.caption("Supports multiple PDFs")

        st.markdown("---")
        st.markdown("### 🚀 About")
        st.caption("DocuMind AI lets you chat with PDFs using AI.")

    # -------------------------------
    # Upload Section
    # -------------------------------
    st.markdown("""
    <div style="
        padding:20px;
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        margin-bottom:20px;
    ">
    <h4>📂 Upload PDFs</h4>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "",
        type="pdf",
        accept_multiple_files=True
    )

    # -------------------------------
    # Process PDFs
    # -------------------------------
    if uploaded_files and st.session_state.db is None:
        with st.spinner("📄 Processing PDFs..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_path = tmp_file.name

                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            split_docs = text_splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings()
            db = FAISS.from_documents(split_docs, embeddings)

            st.session_state.db = db
            st.success("✅ PDFs processed successfully!")

    # -------------------------------
    # Empty State
    # -------------------------------
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; margin-top:40px; color:#94a3b8;">
            <h3>👋 Welcome to DocuMind</h3>
            <p>Upload your PDFs and start asking questions</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Display Chat
    # -------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------------------
    # Chat Input
    # -------------------------------
    st.markdown("### 💬 Chat")

    query = st.chat_input("💬 Ask anything about your documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)

        if st.session_state.db is None:
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("⚠️ Please upload a PDF first.")

        else:
            retriever = st.session_state.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )

            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs[:4]])

            history = "\n".join([
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-6:]
            ])

            template = """
You are an expert AI assistant.

Answer ONLY using the context.
If not found, say: "I couldn't find this in the document".

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

            prompt = ChatPromptTemplate.from_template(template).format(
                history=history,
                context=context,
                question=query
            )

            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""

                try:
                    with st.spinner("🤖 Thinking..."):
                        for chunk in llm.stream(prompt):
                            if hasattr(chunk, "content"):
                                full_response += chunk.content
                                response_placeholder.markdown(full_response + " ▌")

                    if not full_response.strip():
                        full_response = "⚠️ No meaningful answer found in document."

                    response_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"❌ Error: {str(e)}"
                    st.error(full_response)

            # Sources
            with st.expander("📄 Sources"):
                for i, doc in enumerate(docs):
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"**Source {i+1} (Page {page})**")
                    st.write(doc.page_content[:300] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

    # -------------------------------
    # Footer
    # -------------------------------
    st.markdown("""
    <hr style="margin-top:50px; opacity:0.2;">
    <p style='text-align:center; color:#64748b; font-size:14px;'>
    ⚡ Powered by AI • Built with ❤️ by Parth
    </p>
    """, unsafe_allow_html=True)
