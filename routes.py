from fastapi import APIRouter
from schemas import ChatRequest, ChatResponse
from services import run_chatbot

router = APIRouter(prefix="/chat", tags=["Chatbot"])

@router.post("/", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    reply = run_chatbot(
        session_id=request.session_id,
        user_message=request.message
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=reply
    )
