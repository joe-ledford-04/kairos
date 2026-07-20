from fastapi import APIRouter, Depends
from service.schemas.chat import ChatRequest, ChatResponse
from service.services.chat_service import generate_chat_answer
from service.dependencies import get_current_user

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"]
)

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
) -> ChatResponse:
    answer = await generate_chat_answer(current_user, request.message, request.static)
    return ChatResponse(answer=answer)