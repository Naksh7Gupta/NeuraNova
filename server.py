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
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Health check
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
    
async def chatbot_stream_generator(session_id: str, user_message: str):
    """
    Simulate streaming from run_chatbot.
    Replace this with real LLM streaming if supported.
    """
    full_reply = run_chatbot(session_id=session_id, user_message=user_message)
    for word in full_reply.split():
        yield word + " "
        await asyncio.sleep(0.02) 
    yield "\n" 

# ----------------- STREAMING ENDPOINT -----------------
@app.post("/chat_stream/")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        chatbot_stream_generator(request.session_id, request.message),
        media_type="text/plain"
    )

@app.post("/get_chat")
def get_chat_endpoint(request: GetChat):
    return get_chats(request)

@app.post("/upload/")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = os.path.join(UPLOAD_DIR, session_id)
    
    file_path = file_path + ".pdf"

    # Save PDF
    with open(file_path, "wb") as f:
        f.write(await file.read())

    file_info = {
        "original_name": file.filename,
        "content_type": file.content_type,
        "path": file_path,
    }

    return {
        "session_id": session_id,
        "status": "📄 File uploaded successfully",
        "file": file_info
    }
