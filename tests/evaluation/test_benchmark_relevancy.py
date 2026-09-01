"""DeepEval Answer Relevancy and Multilingual Benchmark for ShasanAI.

Evaluates semantic answer relevancy across English, Hindi, and Hinglish administrative queries.
"""

import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from tests.evaluation.custom_eval_models import LocalOllamaEvalLLM


@pytest.mark.asyncio
async def test_relevancy_english_administrative_query():
    """Verify answer relevancy for direct English administrative inquiries."""
    eval_model = LocalOllamaEvalLLM()

    input_query = "What are the rules regarding annual transfers in the Forest Department in Uttarakhand?"
    actual_output = (
        "Under GO-1345/XII/2018 issued by the Forest Department, inter-district transfer requests "
        "must be processed exclusively during the annual transfer window in May."
    )

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=eval_model, include_reason=False)
    await metric.a_measure(test_case)

    assert metric.score >= 0.7, f"Answer Relevancy score {metric.score} below threshold. Reason: {metric.reason}"


@pytest.mark.asyncio
async def test_relevancy_hinglish_administrative_query():
    """Verify answer relevancy for mixed Hindi-English (Hinglish) administrative queries."""
    eval_model = LocalOllamaEvalLLM()

    input_query = "Forest department me transfer ka time kab hota hai as per GO?"
    actual_output = (
        "GO-1345/XII/2018 ke anusar Forest department me inter-district transfer requests "
        "har saal keval May ke mahine me process ki jayengi."
    )

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
    )

    metric = AnswerRelevancyMetric(threshold=0.6, model=eval_model, include_reason=False)
    await metric.a_measure(test_case)

    assert metric.score >= 0.6, f"Hinglish Relevancy score {metric.score} below threshold. Reason: {metric.reason}"
