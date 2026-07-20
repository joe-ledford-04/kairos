from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    static: bool = Field(default=True)

class ChatResponse(BaseModel):
    answer: str