import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    model_id: str
    backend: str
    host: str
    port: int
    max_new_tokens: int
    temperature: float
    top_p: float
    trust_remote_code: bool
    tensor_parallel_size: int
    gpu_memory_utilization: float
    quantization: str
    quantization_bits: int
    enable_guardrails: bool
    max_prompt_chars: int
    max_request_new_tokens: int
    blocked_terms: str



def get_settings() -> Settings:
    return Settings(
        model_id=os.getenv("MODEL_ID", "gpt2"),
        backend=os.getenv("INFERENCE_BACKEND", "vllm").lower(),
        host=os.getenv("HOST", "0.0.0.0"),
        port=_to_int(os.getenv("PORT"), 8000),
        max_new_tokens=_to_int(os.getenv("MAX_NEW_TOKENS"), 128),
        temperature=_to_float(os.getenv("TEMPERATURE"), 0.7),
        top_p=_to_float(os.getenv("TOP_P"), 0.95),
        trust_remote_code=_to_bool(os.getenv("TRUST_REMOTE_CODE"), False),
        tensor_parallel_size=_to_int(os.getenv("TENSOR_PARALLEL_SIZE"), 1),
        gpu_memory_utilization=_to_float(os.getenv("GPU_MEMORY_UTILIZATION"), 0.9),
        quantization=os.getenv("QUANTIZATION", "none").lower(),
        quantization_bits=_to_int(os.getenv("QUANTIZATION_BITS"), 0),
        enable_guardrails=_to_bool(os.getenv("ENABLE_GUARDRAILS"), True),
        max_prompt_chars=_to_int(os.getenv("MAX_PROMPT_CHARS"), 4000),
        max_request_new_tokens=_to_int(os.getenv("MAX_REQUEST_NEW_TOKENS"), 512),
        blocked_terms=os.getenv("BLOCKED_TERMS", ""),
    )
