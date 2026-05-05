import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

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
# Upload PDFs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
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

        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(split_docs, embeddings)

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
if prompt := st.chat_input("💬 Ask something about your documents..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

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

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

        # Retrieve relevant chunks
        docs = retriever.invoke(prompt)

        # Limit context
        context = "\n\n".join([
            doc.page_content for doc in docs[:3]
        ])

        # Chat history
        history_text = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in st.session_state.chat_history
        ])

        # Prompt
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
                    if hasattr(chunk, "content"):
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")

                if not full_response.strip():
                    full_response = "⚠️ No answer found in document."

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                st.error(full_response)

        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        st.session_state.chat_history.append({
            "user": prompt,
            "assistant": full_response
        })

        # -------------------------------
        # Sources
        # -------------------------------
        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i+1} (Page {page})**")
                st.write(doc.page_content[:300] + "...")
