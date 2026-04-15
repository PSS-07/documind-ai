import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.title("📘 AI Study Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_kwargs={"k":3})

    llm = Ollama(model="mistral")

    query = st.text_input("Ask a question")

    if query:
        # Retrieve relevant chunks
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based ONLY on the context below.

        Context:
        {context}

        Question:
        {question}
        """)

        final_prompt = prompt.format(context=context, question=query)

        response = llm.invoke(final_prompt)

        st.write(response)
