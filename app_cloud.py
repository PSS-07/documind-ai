import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="DocuMind AI", layout="wide")
st.title("📘 DocuMind AI")
st.caption("💡 Chat with your PDFs using AI")

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

# -------------------------------
# Clear Chat
# -------------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# -------------------------------
# Process PDF (ONCE)
# -------------------------------
if uploaded_file and st.session_state.db is None:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 🔥 Smart Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(split_docs, embeddings)

        st.session_state.db = db
        st.success("✅ PDF processed!")

# -------------------------------
# Display Chat
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Chat Input
# -------------------------------
if prompt := st.chat_input("Ask something..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.db is None:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload a PDF first.")
    else:
        retriever = st.session_state.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

        # 🔍 Retrieve relevant chunks
        docs = retriever.invoke(prompt)

        # 🔥 Context Filtering (limit size)
        context = "\n\n".join([doc.page_content for doc in docs[:3]])

        # 🧠 Format chat history
        history_text = "\n".join(
            [f"User: {msg['user']}\nAssistant: {msg['assistant']}"
             for msg in st.session_state.chat_history]
        )

        # 🔥 Strong Prompt
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
        # Streaming Response
        # -------------------------------
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in llm.stream(final_prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                st.error(full_response)

        # Save history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        st.session_state.chat_history.append({
            "user": prompt,
            "assistant": full_response
        })
