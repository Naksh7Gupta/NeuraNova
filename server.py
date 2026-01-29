from fastapi import FastAPI, UploadFile, Form,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from routes import router as chat_router
from schemas import ChatRequest, ChatResponse, GetChat
from services import run_chatbot, get_chats
import os
import asyncio


app = FastAPI(
    title="LangGraph Chatbot API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neuranova-ai.vercel.app/"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "Backend running 🚀"}

# Chat endpoint
app.include_router(chat_router)

# File upload endpoint
@app.post("/chat/", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    reply = run_chatbot(
        session_id=request.session_id,
        user_message=request.message
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=reply
    )

@app.post("/get_chat")
def get_chat_endpoint(request: GetChat):
    return get_chats(request)
