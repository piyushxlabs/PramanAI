import os
from typing import Any, Optional, Union
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()


class LocalOllamaEvalLLM(DeepEvalBaseLLM):
    """DeepEval evaluator LLM running strictly against local/configured Ollama with stable JSON extraction."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None, *args: Any, **kwargs: Any) -> None:
        self.model_name = model_name or os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:14b"
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = 120.0
        super().__init__(model=self.model_name, *args, **kwargs)

    def load_model(self) -> Any:
        """Initializes ChatOllama instance with configured model, base URL, and keep_alive."""
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,
            timeout=self.timeout,
            keep_alive="24h",
            num_ctx=8192,
        )

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None, *args: Any, **kwargs: Any) -> Union[str, BaseModel]:
        """Synchronously invokes local Ollama."""
        llm = self.load_model()
        if schema:
            instruction = f"\n\nYou must respond STRICTLY with a valid JSON object adhering to this schema:\n{schema.model_json_schema()}"
            res = llm.invoke([HumanMessage(content=prompt + instruction)])
            text = res.content if hasattr(res, "content") else str(res)
            try:
                # Extract first valid json object if enclosed in markdown
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                parsed = json.loads(text)
                return schema.model_validate(parsed)
            except Exception:
                return schema.model_validate(json.loads(text.strip()))

        res = llm.invoke([HumanMessage(content=prompt)])
        return res.content if hasattr(res, "content") else str(res)

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None, *args: Any, **kwargs: Any) -> Union[str, BaseModel]:
        """Asynchronously invokes local Ollama."""
        llm = self.load_model()
        if schema:
            instruction = f"\n\nYou must respond STRICTLY with a valid JSON object adhering to this schema:\n{schema.model_json_schema()}"
            res = await llm.ainvoke([HumanMessage(content=prompt + instruction)])
            text = res.content if hasattr(res, "content") else str(res)
            try:
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                parsed = json.loads(text)
                return schema.model_validate(parsed)
            except Exception:
                return schema.model_validate(json.loads(text.strip()))

        res = await llm.ainvoke([HumanMessage(content=prompt)])
        return res.content if hasattr(res, "content") else str(res)
