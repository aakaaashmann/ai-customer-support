import os
import json
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

# --- LangChain & AI Imports ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
app = FastAPI()

# MIDDLEWARE SECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store: Dict[str, List] = {} 

# --- API KEY CHECKS ---
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file")

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Please add it to .env or Render Environment Variables.")

def load_faq_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    documents = []
    for item in data:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"
        documents.append(Document(page_content=content))
    return documents

print("Loading AI models via API (Low RAM mode)...")

# --- EMBEDDING MODEL ---
# Using the API instead of downloading to RAM
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LOAD FAQ DATA
try:
    faq_documents = load_faq_data("data/faq_data.json") 
except FileNotFoundError:
    # Fallback if running from inside app folder
    faq_documents = load_faq_data("../data/faq_data.json")

# --- UPDATED RETRY LOGIC (MORE PATIENCE) ---
print("Generating vector store... (This depends on Hugging Face API speed)")
vector_store = None

# Increase attempts to 10, and wait 30 seconds between tries.
# This gives the model 5 minutes to wake up.
for attempt in range(10): 
    try:
        # 1. Test the connection first
        print(f"Attempt {attempt+1}/10: Connecting to Hugging Face...")
        test_embed = embeddings.embed_query("test connection")
        
        # 2. If test passes, build the store
        vector_store = FAISS.from_documents(faq_documents, embeddings)
        print("Success: Vector store created!")
        break 
    except Exception as e:
        print(f"Attempt {attempt+1} failed. Error: {e}")
        print("The Hugging Face model is likely loading. Waiting 30 seconds...")
        time.sleep(30)

if vector_store is None:
    # If it fails after 5 minutes, we cannot start.
    print("CRITICAL ERROR: Could not connect to Embedding API.")
    raise ValueError("Failed to initialize Vector Store. Check Render Logs for specific error details.")

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# LLM SETUP
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# PROMPT SETUP
system_prompt = """You are a customer support agent.
Use the Context below to answer the user.
If the answer is NOT in the context, reply exactly: "ESCALATION_REQUIRED".

Context: 
{context} 
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user", "{question}")
])

class ChatRequest(BaseModel):
    session_id: str 
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query

    if session_id not in session_store:
        session_store[session_id] = [] 
    
    history = session_store[session_id]

    # RAG Logic
    docs = retriever.invoke(user_query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt_template | llm
    response_message = chain.invoke({
        "context": context_text,
        "chat_history": history, 
        "question": user_query
    })
    
    bot_reply = response_message.content

    # Update History
    history.append(HumanMessage(content=user_query))
    history.append(AIMessage(content=bot_reply))
    session_store[session_id] = history

    return {"response": bot_reply, "session_id": session_id}