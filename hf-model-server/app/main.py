from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.engine import GenerationOptions, ModelEngine

settings = get_settings()
engine = ModelEngine(settings)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    model_id: str
    backend: str
    generated_text: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.load()
    yield


app = FastAPI(title="HF Model Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_id": settings.model_id,
        "backend": settings.backend,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    try:
        options = GenerationOptions(
            max_new_tokens=payload.max_new_tokens or settings.max_new_tokens,
            temperature=payload.temperature if payload.temperature is not None else settings.temperature,
            top_p=payload.top_p if payload.top_p is not None else settings.top_p,
        )
        output_text = engine.generate(payload.prompt, options)
        return GenerateResponse(
            model_id=settings.model_id,
            backend=settings.backend,
            generated_text=output_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
