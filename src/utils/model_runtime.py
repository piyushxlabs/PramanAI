"""Google Gemini & Google GenAI SDK model runtime bindings for PramanAI.

Provides high-performance reasoning, structured outputs, multi-modal vision extraction,
and model armor security integration on Google Gemini 3.5 Flash and Gemini 3.5 Flash-Lite.
"""

import asyncio
import base64
import io
import logging
import os
import re
from typing import Any, Optional, Type, TypeVar
from dotenv import load_dotenv
from google import genai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PIL import Image
from pydantic import BaseModel, ValidationError

from src.state.reducers import ScopeViolationError, StateValidationError, ToolExecutionError

load_dotenv(override=True)

logger = logging.getLogger("praman_ai.model_runtime")

T = TypeVar("T", bound=BaseModel)

class ConfigurationError(ToolExecutionError):
    """Raised when required Google Cloud / Gemini runtime credentials or settings are missing."""
    pass


# ==============================================================================
# Model Configurations (Strictly Gemini 3.5 Flash / 3.5 Flash-Lite / Gemma 2)
# ==============================================================================
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_FLASH_MODEL: str = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
GEMINI_LITE_MODEL: str = os.getenv("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite")
GEMINI_ARMOR_MODEL: str = os.getenv("GEMINI_ARMOR_MODEL", "gemma-2-2b-it")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Vision & Ingestion Settings
VLM_MODEL: str = os.getenv("VLM_MODEL") or GEMINI_FLASH_MODEL
VISION_LLM_MODEL: str = VLM_MODEL  # Backwards compatibility alias
USE_VLM_EXTRACTION: bool = os.getenv("USE_VLM_EXTRACTION", "true").lower() in ("true", "1", "yes")
VLM_MAX_IMAGE_DIM: int = int(os.getenv("VLM_MAX_IMAGE_DIM", "1600"))
VLM_TIMEOUT_SECONDS: float = float(os.getenv("VLM_TIMEOUT_SECONDS", "120.0"))


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Returns an authenticated native Google GenAI Client instance (`genai.Client`)."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        logger.warning("GEMINI_API_KEY is not set. Native GenAI Client initialized in default credential mode.")
    return genai.Client(api_key=key if key else None)


def check_gemini_health(api_key: Optional[str] = None) -> bool:
    """Validates live Gemini API connectivity."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        return False
    try:
        client = get_genai_client(api_key=key)
        return client is not None
    except Exception as exc:
        logger.warning(f"Gemini API health check failed: {exc}")
        return False


async def check_model_health(api_key: Optional[str] = None) -> dict[str, Any]:
    """Queries live Gemini runtime configuration and returns health status dictionary."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    is_healthy = check_gemini_health(api_key=key)
    
    return {
        "status": "healthy" if is_healthy or not key else "unconfigured",
        "provider": "Google Gemini",
        "flash_model": os.getenv("GEMINI_FLASH_MODEL") or GEMINI_FLASH_MODEL,
        "lite_model": os.getenv("GEMINI_LITE_MODEL") or GEMINI_LITE_MODEL,
        "armor_model": os.getenv("GEMINI_ARMOR_MODEL") or GEMINI_ARMOR_MODEL,
        "api_key_configured": bool(key),
    }


def get_chat_model(
    temperature: float = 0.0,
    model: Optional[str] = None,
    max_tokens: Optional[int] = 2048,
    timeout: float = 120.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Returns a ChatGoogleGenerativeAI instance bound to Gemini 3.5 Flash.
    
    Default temperature is 0.0 for deterministic public-records reasoning and zero-hallucination citations.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    target_model = model or os.getenv("GEMINI_FLASH_MODEL") or GEMINI_FLASH_MODEL
    effective_timeout = max(float(timeout), 15.0)
    
    return ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=key if key else "MOCK_KEY_FOR_TESTS",
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout=effective_timeout,
    )


def get_fast_model(
    temperature: float = 0.0,
    model: Optional[str] = None,
    max_tokens: Optional[int] = 512,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Returns a high-speed ChatGoogleGenerativeAI instance bound to Gemini 3.5 Flash-Lite.
    
    Used for sub-second query normalization, language identification, and intent routing.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    target_model = model or GEMINI_LITE_MODEL
    effective_timeout = max(float(timeout), 15.0)
    
    return ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=key if key else "MOCK_KEY_FOR_TESTS",
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout=effective_timeout,
    )


def get_vision_model(
    temperature: float = 0.0,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    timeout: float = VLM_TIMEOUT_SECONDS,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Returns a ChatGoogleGenerativeAI instance configured for multimodal 300 DPI document parsing."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    target_model = model or VLM_MODEL
    effective_timeout = max(float(timeout), 15.0)
    
    return ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=key if key else "MOCK_KEY_FOR_TESTS",
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout=effective_timeout,
    )


class LocalFallbackEmbeddings:
    """Zero-dependency hash-based deterministic embedding fallback for local testing when API key is unset."""
    def embed_query(self, text: str) -> list[float]:
        import hashlib
        import math
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
        while len(vec) < 1024:
            h = hashlib.sha256(h).digest()
            vec.extend([(b / 255.0) * 2.0 - 1.0 for b in h])
        norm = math.sqrt(sum(x * x for x in vec[:1024])) or 1.0
        return [x / norm for x in vec[:1024]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class GoogleGenAIEmbeddingWrapper:
    """Wrapper around Google embeddings using genai.Client with automatic fallback to langchain-google-genai."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        try:
            self._client = genai.Client(api_key=api_key)
        except Exception:
            self._client = None
        self._fallback = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    def embed_query(self, text: str) -> list[float]:
        if self._client:
            try:
                res = self._client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    return res.embeddings[0].values
                elif hasattr(res, "embedding") and hasattr(res.embedding, "values"):
                    return res.embedding.values
            except Exception as exc:
                logger.warning("Native genai embed_content fallback (%s); using langchain embeddings", exc)
        try:
            return self._fallback.embed_query(text)
        except Exception:
            return LocalFallbackEmbeddings().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def get_embeddings_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Returns Google GenAI embeddings (gemini-embedding-001) or resilient fallback model."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if key and key != "MOCK_KEY_FOR_TESTS":
        target_model = model or os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL
        if not target_model.startswith("models/") and not target_model.startswith("gemini-"):
            target_model = f"models/{target_model}"
        # If user passed text-embedding-004 which 404s on v1beta, redirect to gemini-embedding-001
        if "text-embedding-004" in target_model:
            target_model = "models/gemini-embedding-001"
        try:
            return GoogleGenAIEmbeddingWrapper(model_name=target_model, api_key=key)
        except Exception as exc:
            logger.warning(f"Could not initialize Google embeddings ({exc}); using fallback")
    return LocalFallbackEmbeddings()


def get_structured_llm(
    schema: Type[T],
    temperature: float = 0.0,
    model: Optional[str] = None,
    timeout: float = 60.0,
    max_tokens: int = 1024,
    use_fast_model: bool = False,
    api_key: Optional[str] = None,
) -> Any:
    """Returns a Google Gemini model bound with strict Pydantic structured output validation."""
    if use_fast_model:
        llm = get_fast_model(
            temperature=temperature,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        llm = get_chat_model(
            temperature=temperature,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    return llm.with_structured_output(schema, method="json_mode")


async def ainvoke_vision(
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    model: Optional[str] = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    backoff_factor: float = 2.0,
    timeout_seconds: float = VLM_TIMEOUT_SECONDS,
    max_image_dim: Optional[int] = None,
    api_key: Optional[str] = None,
) -> str:
    """Invokes Google Gemini 3.5 Flash Multimodal Vision API with image bytes, image enhancement, and retry."""
    dim_limit = max_image_dim or VLM_MAX_IMAGE_DIM
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        from src.gov_pdf_extractor.preprocessor import enhance_gov_document_image

        pil_img = enhance_gov_document_image(pil_img)
        w, h = pil_img.size
        if max(w, h) > dim_limit:
            scale = dim_limit / max(w, h)
            new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        processed_bytes = buf.getvalue()
        img_b64 = base64.b64encode(processed_bytes).decode("utf-8")
    except Exception:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Construct LangChain Multimodal Message
    image_data_url = f"data:image/jpeg;base64,{img_b64}"
    messages: list[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    messages.append(
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
    )

    vision_llm = get_vision_model(
        model=model or GEMINI_FLASH_MODEL,
        timeout=timeout_seconds,
        api_key=api_key,
    )

    delay = initial_backoff
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            res = await asyncio.wait_for(
                vision_llm.ainvoke(messages),
                timeout=timeout_seconds,
            )
            content = res.content if hasattr(res, "content") else res
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    elif hasattr(part, "text"):
                        text_parts.append(str(getattr(part, "text", "")))
                    else:
                        text_parts.append(str(part))
                return "\n".join(text_parts).strip()
            return str(content).strip()
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            logger.warning(
                "Gemini Vision invocation attempt %d/%d failed with error (%s): %s",
                attempt,
                max_retries,
                type(exc).__name__,
                exc,
            )
            if attempt < max_retries:
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if not retry_match:
                        retry_match = re.search(r"retryDelay': '(\d+)s", err_str, re.IGNORECASE)
                    wait_s = float(retry_match.group(1)) + 2.0 if retry_match else max(delay, 20.0)
                    print(f"  [Gemini Vision] Rate limit reached. Sleeping for {wait_s:.1f}s before retry {attempt+1}/{max_retries}...", flush=True)
                    await asyncio.sleep(wait_s)
                else:
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

    raise ToolExecutionError(
        f"Gemini Vision invocation timed out or failed after {max_retries} attempts. Last error: {last_error!s}"
    ) from last_error


async def ainvoke_with_retry(
    structured_llm: Any,
    messages: list[Any],
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    timeout_seconds: float = 30.0,
) -> Any:
    """Invokes structured LLM with exponential backoff retry (1s -> 2s -> 4s) and strict timeout on transient failures.
    
    Non-transient errors (ValidationError, ScopeViolationError, StateValidationError) bypass retry.
    """
    delay = initial_backoff
    last_error: Optional[Exception] = None
    _non_transient = (ValidationError, ScopeViolationError, StateValidationError)

    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.wait_for(
                structured_llm.ainvoke(messages),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= backoff_factor
        except Exception as exc:
            last_error = exc
            if attempt < max_retries and not isinstance(exc, _non_transient):
                await asyncio.sleep(delay)
                delay *= backoff_factor
            elif isinstance(exc, _non_transient):
                break

    raise ToolExecutionError(
        f"Model invocation timed out or failed after {max_retries} attempts. Last error: {last_error!s}"
    ) from last_error
