import ollama
import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

TIMEOUT = 60  # seconds per LLM call


class OllamaService:
    def __init__(self, model: str = "llama2", host: str = None):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=self.host, timeout=TIMEOUT)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(self, prompt: str, system_prompt: Optional[str] = None, model: str = "llama2") -> str:
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use only the information from the context to answer questions. "
                "If the context doesn't contain enough information to answer the question, say so politely."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(model=model or self.model, messages=messages)
        return response["message"]["content"]

    def list_models(self) -> list:
        try:
            models = self._client.list()
            return [m["name"] for m in models["models"]]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        try:
            self._client.pull(model_name)
            return True
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False

    def set_model(self, model: str) -> None:
        self.model = model
