from pydantic import BaseModel
from typing import Optional
from fastapi import UploadFile

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

class GetChat(BaseModel):
    session_id: str
    
class ChatImageRequest(BaseModel):
    session_id: str
    message: Optional[str] = ""
    image: UploadFile