"""Verification tests for Step 2: Local Model Bindings & Smoke Test (Ollama qwen2.5:7b + bge-m3)."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from src.state.schema import QueryInterpretation, ScopeScreenDecision
from src.utils.model_runtime import (
    check_ollama_health,
    get_chat_model,
    get_embeddings_model,
    get_structured_llm,
)

NODE1_SYSTEM_PROMPT = """<identity_and_role>
You are the Query Interpretation node of the GO-Retrieval & Citation Agent, running as Node 1 of a LangGraph StateGraph on qwen2.5:7b.
Your purpose is to parse the officer's raw query into a normalized, typed form.
</identity_and_role>
<output_formatting_rules>
Respond only with a single QueryInterpretation JSON object with fields:
- query_text: string
- query_language: "hi" | "en" | "hinglish"
- query_filters: {"department": string | null, "year_range": [int, int] | null, "policy_category": string | null}
</output_formatting_rules>"""


@pytest.mark.asyncio
async def test_ollama_health_and_models_available():
    """Verify local Ollama daemon connectivity and model availability (qwen2.5 & bge-m3)."""
    import os
    health = await check_ollama_health()
    assert health["status"] == "healthy"
    expected_model = os.getenv("LLM_MODEL", "qwen2.5:14b")
    assert expected_model in health["llm_model"] or "qwen2.5" in health["llm_model"]
    assert "bge-m3" in health["embedding_model"]
    assert len(health["available_models"]) >= 2


@pytest.mark.asyncio
async def test_chat_ollama_smoke_inference():
    """Verify local qwen2.5:7b smoke inference invocation via async ainvoke."""
    llm = get_chat_model(temperature=0.0)
    response = await llm.ainvoke([{"role": "user", "content": "Respond with single word: OK"}])
    assert response is not None
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0


@pytest.mark.asyncio
async def test_structured_output_json_mode():
    """Verify native structured output validation in JSON mode using Pydantic V2 model."""
    structured_llm = get_structured_llm(QueryInterpretation, temperature=0.0)
    messages = [
        SystemMessage(content=NODE1_SYSTEM_PROMPT),
        HumanMessage(content="2018 mein forest department ke transfer policy ka GO kya hai?"),
    ]
    result = await structured_llm.ainvoke(messages)
    assert isinstance(result, QueryInterpretation)
    assert result.query_language in ("hi", "en", "hinglish")
    assert result.query_text is not None
    assert result.query_filters is not None


@pytest.mark.asyncio
async def test_bge_m3_embeddings_dimensions():
    """Verify bge-m3 embedding generation produces exact 1024-dimensional dense vectors."""
    embeddings = get_embeddings_model()
    vector = await embeddings.aembed_query("Uttarakhand Government Order circular on transfer policy")
    assert isinstance(vector, list)
    assert len(vector) == 1024
    assert all(isinstance(val, float) for val in vector)


def test_temperature_determinism_configuration():
    """Verify that default chat model is configured with temperature=0.0 for zero-variance reasoning."""
    llm = get_chat_model()
    assert llm.temperature == 0.0
