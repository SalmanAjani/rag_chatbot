from fastapi import APIRouter

from src.api.chat_schema import ChatRequest, ChatResponse
from src.chat.rag import ask_question

router = APIRouter()


@router.post("/ask")
def ask(request: ChatRequest):
    answer = ask_question(request.question)

    return ChatResponse(answer=answer)
