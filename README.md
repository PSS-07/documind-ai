🚀 DocuMind AI

DocuMind AI is an AI-powered study assistant that allows users to upload PDF documents and interact with them using natural language.

It uses a Retrieval-Augmented Generation (RAG) pipeline with a local LLM (Mistral) to generate accurate, context-aware answers directly from documents.

🔥 Features
📄 Upload and process PDF documents
💬 Chat with your documents (ChatGPT-style UI)
⚡ Fast retrieval using FAISS vector database
🧠 Context-aware answers using RAG
🔒 Fully local LLM support via Ollama (Mistral)
🚀 Streaming responses for real-time interaction
🛠️ Tech Stack
Python
Streamlit
LangChain
FAISS (Vector Database)
HuggingFace Embeddings
Ollama (Mistral LLM)
🧠 How It Works
PDF is uploaded and parsed
Text is converted into embeddings
Stored in FAISS vector database
User query is embedded and matched
Relevant chunks are retrieved
Mistral generates the final answer
📸 Demo

Add a screenshot here

⚡ Setup Instructions
git clone https://github.com/PSS-07/documind-ai.git
cd documind-ai

python3 -m venv ai-saas-env
source ai-saas-env/bin/activate

pip install -r requirements.txt
streamlit run app.py
🚀 Future Improvements
Multi-PDF support
Chat memory (context-aware conversations)
User authentication (SaaS features)
Cloud deployment
💡 Use Cases
Study assistant for exams
Research paper analysis
Document Q&A system
👨‍💻 Author

Parth Sharma
