import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# PREMIUM UI STYLING
# -------------------------------
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease-in-out;
}

[data-testid="stChatMessage"]:hover {
    transform: scale(1.01);
}

/* User message */
[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]:first-child) {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
}

/* Assistant message */
[data-testid="stChatMessage"]:not(:has(div[data-testid="stMarkdownContainer"]:first-child)) {
    background: rgba(255, 255, 255, 0.05);
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #3b82f6;
    border-radius: 15px;
    padding: 20px;
}

/* Input */
[data-testid="stChatInput"] {
    border-radius: 20px;
    background-color: #1e293b;
}

/* Buttons */
button {
    border-radius: 12px !important;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HERO SECTION
# -------------------------------
st.markdown("""
<div style="text-align:center; padding: 20px 0;">
    <h1 style="font-size: 42px;">📘 DocuMind AI</h1>
    <p style="color: #94a3b8; font-size:18px;">
        Chat with your documents like never before 🚀
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Upload PDFs and start chatting")

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.markdown("### 📂 Upload your PDFs")

uploaded_files = st.file_uploader(
    "Drag & drop or browse",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------------
# PROCESS PDFs
# -------------------------------
if uploaded_files and st.session_state.db is None:
    with st.spinner("🔄 Processing PDFs..."):

        all_docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(split_docs, embeddings)

        st.session_state.db = db
        st.success("✅ PDFs processed successfully!")

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# CHAT INPUT
# -------------------------------
if st.session_state.db is None:
    prompt = st.chat_input("Upload PDFs first...", disabled=True)
else:
    prompt = st.chat_input("💬 Ask something about your documents...")

# -------------------------------
# CHAT LOGIC
# -------------------------------
if prompt:

    # USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    retriever = st.session_state.db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    docs = retriever.invoke(prompt)

    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    history_text = "\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.chat_history
    ])

    template = """
You are an intelligent AI assistant.

Answer strictly based on the provided context and conversation history.

Rules:
- Do NOT make up answers
- If unsure, say "I don't know based on the document"
- Keep answers clear and structured
- Use bullet points when helpful

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""

    prompt_template = ChatPromptTemplate.from_template(template)

    final_prompt = prompt_template.format(
        chat_history=history_text,
        context=context,
        question=prompt
    )

    # -------------------------------
    # ASSISTANT RESPONSE
    # -------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking... 🤖"):
            try:
                for chunk in llm.stream(final_prompt):
                    if hasattr(chunk, "content"):
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌")

                if not full_response.strip():
                    full_response = "⚠️ No answer found in document."

                placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                st.error(full_response)

    # SAVE HISTORY
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    st.session_state.chat_history.append({
        "user": prompt,
        "assistant": full_response
    })

    # -------------------------------
    # SOURCES
    # -------------------------------
    with st.expander("📄 Sources"):
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "N/A")
            st.markdown(f"**Source {i+1} (Page {page})**")
            st.write(doc.page_content[:300] + "...")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Built with ❤️ using AI • DocuMind
</p>
""", unsafe_allow_html=True)
