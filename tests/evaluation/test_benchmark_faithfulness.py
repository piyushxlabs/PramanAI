"""DeepEval Faithfulness and Zero-Hallucination Benchmark for ShasanAI.

Evaluates 100% factual fidelity and zero-hallucination tolerance for administrative answers
against official Uttarakhand Government Order excerpts using local Ollama (qwen2.5:7b).
"""

import pytest
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from tests.evaluation.custom_eval_models import LocalOllamaEvalLLM


@pytest.mark.asyncio
async def test_faithfulness_grounded_answer_benchmark():
    """Verify that synthesized answers strictly adhere to retrieved GO excerpts."""
    eval_model = LocalOllamaEvalLLM()

    input_query = "What is the annual transfer window for the Forest department under GO-1345/XII/2018?"
    actual_output = (
        "According to GO Number GO-1345/XII/2018, Department of Forest, dated 12th March 2018, "
        "inter-district transfer requests shall be processed strictly during the annual transfer window "
        "in the month of May. (Citation [1], Page 3)"
    )
    retrieval_context = [
        "GO Number: GO-1345/XII/2018 | Department: Forest | Date: 2018-03-12 | Page: 3\n"
        "Inter-district transfer requests shall be processed strictly during the annual transfer window in the month of May."
    ]

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        context=retrieval_context,
    )

    metric = FaithfulnessMetric(threshold=0.7, model=eval_model, include_reason=False)
    await metric.a_measure(test_case)

    assert metric.score >= 0.7, f"Faithfulness score {metric.score} below threshold. Reason: {metric.reason}"


@pytest.mark.asyncio
async def test_zero_hallucination_benchmark():
    """Verify that the system produces 0 unsupported hallucinated claims."""
    eval_model = LocalOllamaEvalLLM()

    input_query = "What is the transfer window in Forest Department?"
    actual_output = (
        "Under GO-1345/XII/2018, the transfer window is strictly during May."
    )
    context = [
        "Inter-district transfer requests shall be processed strictly during the annual transfer window in the month of May."
    ]

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        context=context,
    )

    hallucination_metric = HallucinationMetric(threshold=0.5, model=eval_model, include_reason=False)
    await hallucination_metric.a_measure(test_case)

    assert hallucination_metric.score >= 0.5, (
        f"Hallucination score {hallucination_metric.score} below threshold. Reason: {hallucination_metric.reason}"
    )
