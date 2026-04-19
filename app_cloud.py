import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import tempfile

# -------------------------------
# Page Config + UI
# -------------------------------
st.set_page_config(page_title="DocuMind AI", layout="wide")

st.markdown("""
# 📘 DocuMind AI
### 💬 Chat with your documents intelligently
""")

st.divider()

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# -------------------------------
# Sidebar (UI polish)
# -------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Upload PDFs and start chatting")

# -------------------------------
# Upload PDFs
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDFs",
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
# Display Chat
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Chat Input
# -------------------------------
query = st.chat_input("Ask a question about your PDFs...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.db is None:
        with st.chat_message("assistant"):
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

        # 🔍 Retrieve docs
        docs = retriever.invoke(query)

        # 🔥 Build context
        context = "\n\n".join([doc.page_content for doc in docs[:4]])

        # 🧠 Chat history (last 5 messages)
        history = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in st.session_state.messages[-6:]
        ])

        # -------------------------------
        # Improved Prompt
        # -------------------------------
        template = """
You are an expert AI assistant for analyzing documents.

Guidelines:
- Answer ONLY using the provided context
- If answer is not in context, say: "I couldn't find this in the document"
- Keep answers clear, structured, and concise
- Use bullet points when helpful
- Explain concepts simply when needed

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

        # -------------------------------
        # Generate Response
        # -------------------------------
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in llm.stream(prompt):
                    if hasattr(chunk, "content"):
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                st.error(full_response)

        # -------------------------------
        # 📄 Source Display (NEW)
        # -------------------------------
        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i+1} (Page {page})**")
                st.write(doc.page_content[:300] + "...")

        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
