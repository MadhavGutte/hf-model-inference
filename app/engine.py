from dataclasses import dataclass
from typing import Optional

from app.config import Settings


@dataclass
class GenerationOptions:
    max_new_tokens: int
    temperature: float
    top_p: float


class ModelEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.backend
        self._llm = None
        self._pipeline = None

    def load(self) -> None:
        if self.backend == "vllm":
            self._load_vllm()
            return
        if self.backend == "transformers":
            self._load_transformers()
            return
        raise ValueError("INFERENCE_BACKEND must be either 'vllm' or 'transformers'")

    def generate(self, prompt: str, options: Optional[GenerationOptions] = None) -> str:
        if options is None:
            options = GenerationOptions(
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
            )

        if self.backend == "vllm":
            return self._generate_vllm(prompt, options)
        if self.backend == "transformers":
            return self._generate_transformers(prompt, options)
        raise ValueError("Unsupported backend")

    def _load_vllm(self) -> None:
        from vllm import LLM

        self._llm = LLM(
            model=self.settings.model_id,
            trust_remote_code=self.settings.trust_remote_code,
            tensor_parallel_size=self.settings.tensor_parallel_size,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
        )

    def _generate_vllm(self, prompt: str, options: GenerationOptions) -> str:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
        )
        outputs = self._llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _load_transformers(self) -> None:
        from transformers import pipeline

        self._pipeline = pipeline(
            task="text-generation",
            model=self.settings.model_id,
            device_map="auto",
            trust_remote_code=self.settings.trust_remote_code,
        )

    def _generate_transformers(self, prompt: str, options: GenerationOptions) -> str:
        result = self._pipeline(
            prompt,
            max_new_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            do_sample=options.temperature > 0,
        )
        text = result[0]["generated_text"]
        if text.startswith(prompt):
            return text[len(prompt) :]
        return text
