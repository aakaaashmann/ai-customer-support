# 🤖 AI Customer Support Agent

## 📖 Project Overview
This project is an intelligent Customer Support Bot designed to automate FAQ responses and handle complex conversational flows. Built as part of the **Learning and Practice** assignment, this agent simulates a real-world support environment.

It utilizes **Retrieval-Augmented Generation (RAG)** to provide accurate answers from a knowledge base and maintains **Contextual Memory** to hold natural, multi-turn conversations.

## ✨ Key Features
* **🧠 Contextual Memory:** Unlike basic bots, this agent remembers previous messages in the conversation, allowing users to ask follow-up questions (e.g., "What did I just ask?").
* **🔍 RAG Powered:** Uses **FAISS** and **HuggingFace Embeddings** to retrieve accurate answers from a custom dataset (`faq_data.json`).
* **🚨 Smart Escalation:** Automatically detects when a user's query falls outside the knowledge base and triggers an `ESCALATION_REQUIRED` response for human intervention.
* **⚡ High-Performance API:** Built on **FastAPI** with REST endpoints to handle concurrent user sessions efficiently.
* **🆔 Session Management:** Capable of tracking distinct user sessions simultaneously, ensuring data isolation between different users.
* **💬 Interactive UI:** Includes a lightweight HTML/JS frontend for real-time testing and demonstration.

## 🛠️ Tech Stack
* **Backend Framework:** FastAPI (Python)
* **LLM Orchestration:** LangChain
* **Model Provider:** Groq (Llama-3-70b-Versatile)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Frontend:** HTML5, CSS3, JavaScript

## ⚙️ Setup & Installation
1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd ai-customer-support
2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Environment Variables: Create a .env file in the root directory:**
   ```bash
   GROQ_API_KEY=gsk_...
5. **Run the Server:**
   ```bash
   uvicorn app.main:app --reload
