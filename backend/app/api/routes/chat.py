from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.chat_service import generate_chat_response


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's sleep-wellness question",
    )


class ChatResponse(BaseModel):
    reply: str


@router.post("", response_model=ChatResponse)
async def chat( request: Request,
    chat_request: ChatRequest,) -> ChatResponse:
    message = chat_request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    knowledge_service = getattr(request.app.state, "knowledge_service", None)
    if knowledge_service is None:
        raise HTTPException(
            status_code=503,
            detail="The sleep knowledge base is unavailable, so the assistant cannot answer right now.",
        )

    try:

        reply = await generate_chat_response(message,knowledge_service,)
        return ChatResponse(reply=reply)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
