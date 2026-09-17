"""LLM-judged quality metrics via DeepEval against the deployed service.

Uses the same Azure OpenAI deployment as judge. These are quality signals, not
hard correctness gates — deterministic tests remain authoritative.
"""

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase

from conftest import GOLDENS, responses  # noqa: F401  (fixture import)

judge = AzureOpenAIModel(
    model="gpt-5-nano",
    deployment_name="gpt-5-nano",
)

ANSWERED = [c for c in GOLDENS if c["expected_intent"] == "policy_question"]


@pytest.mark.parametrize("case", ANSWERED, ids=[c["id"] for c in ANSWERED])
def test_answer_relevancy(case, responses):
    body = responses[case["id"]]["response"]
    test_case = LLMTestCase(input=case["question"], actual_output=body["answer"] or "")
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(test_case)
    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize("case", ANSWERED, ids=[c["id"] for c in ANSWERED])
def test_faithfulness(case, responses):
    body = responses[case["id"]]["response"]
    context = [c["quote"] for c in body["citations"]]
    if not context:
        pytest.skip("no grounded citations to judge faithfulness against")
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=body["answer"] or "",
        retrieval_context=context,
    )
    metric = FaithfulnessMetric(threshold=0.8, model=judge, include_reason=True)
    metric.measure(test_case)
    assert metric.is_successful(), metric.reason
