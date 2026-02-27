from dataclasses import dataclass

from app.config import Settings


class GuardrailViolation(ValueError):
    pass


@dataclass
class Guardrails:
    settings: Settings

    def __post_init__(self) -> None:
        raw_terms = self.settings.blocked_terms or ""
        self.blocked_terms = [
            term.strip().lower() for term in raw_terms.split(",") if term.strip()
        ]

    def validate_prompt(self, prompt: str) -> None:
        if not self.settings.enable_guardrails:
            return

        if len(prompt) > self.settings.max_prompt_chars:
            raise GuardrailViolation(
                f"Prompt exceeds MAX_PROMPT_CHARS={self.settings.max_prompt_chars}"
            )

        lowered_prompt = prompt.lower()
        for term in self.blocked_terms:
            if term in lowered_prompt:
                raise GuardrailViolation(
                    "Prompt contains blocked content based on BLOCKED_TERMS policy"
                )

    def validate_max_new_tokens(self, requested_tokens: int) -> None:
        if not self.settings.enable_guardrails:
            return

        if requested_tokens > self.settings.max_request_new_tokens:
            raise GuardrailViolation(
                f"max_new_tokens exceeds MAX_REQUEST_NEW_TOKENS={self.settings.max_request_new_tokens}"
            )
