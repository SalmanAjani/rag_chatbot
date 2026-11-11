import logging
import math
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.chat.rag import answer_question, fetch_context
from src.evaluation.test import TestQuestion, load_tests
from src.llms.llm import llm

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
            # Examples
            # Example: rank=2, return 1.0/2 = 0.5
            # Example: rank=3, return 1.0/3 = 0.3333333333333333
            # Example: rank=4, return 1.0/4 = 0.25
            # Example: rank=5, return 1.0/5 = 0.2
            # Example: rank=6, return 1.0/6 = 0.16666666666666666
            # Example: rank=7, return 1.0/7 = 0.14285714285714285
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    # [1,0,1,0] => [1,1,0,0]

    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to score (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    retrieved_docs = fetch_context(test.question)

    # Calculate MRR (average across all keywords)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # Calculate nDCG (average across all keywords)
    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords
    ]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Calculate keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (
        (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    generated_answer, retrieved_docs = answer_question(test.question)

    judge_prompt = f"""
    Question:
    {test.question}

    Generated Answer:
    {generated_answer}

    Reference Answer:
    {test.reference_answer}

    Please evaluate the generated answer on three dimensions:
    1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
    2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
    3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

    Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.
    """

    # gpt-oss models often emit free-form text instead of tool calls; use Groq's
    # native JSON Schema structured outputs (strict) instead of function calling.
    judge_llm = llm.with_structured_output(
        AnswerEval, method="json_schema", strict=True
    )
    answer_eval = judge_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
            },
            {"role": "user", "content": judge_prompt},
        ]
    )

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""

    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = evaluate_retrieval(test)
        except Exception:
            logger.exception("Retrieval evaluation failed for test %s", index)
            continue

        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """Evaluate all answers to tests."""

    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = evaluate_answer(test)[0]
        except Exception:
            logger.exception("Answer evaluation failed for test %s", index)
            continue

        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test."""

    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(
        f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}"
    )
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, _retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def run_all_evaluations(limit: int | None = None, retrieval_only: bool = False):
    """Run evaluation for all tests (or the first `limit` tests)."""

    tests = load_tests()
    if limit is not None:
        tests = tests[:limit]

    total = len(tests)
    mode = "retrieval only" if retrieval_only else "retrieval + answer"
    print(f"\nRunning {total} tests ({mode})...\n")

    retrieval_scores: list[RetrievalEval] = []
    answer_scores: list[AnswerEval] = []

    for index, test in enumerate(tests):
        print(f"{'=' * 80}")
        print(f"Test #{index}/{total - 1} | {test.category}")
        print(f"Question: {test.question}")

        try:
            retrieval_result = evaluate_retrieval(test)
            retrieval_scores.append(retrieval_result)
            print(
                f"Retrieval — MRR: {retrieval_result.mrr:.4f} | "
                f"nDCG: {retrieval_result.ndcg:.4f} | "
                f"Coverage: {retrieval_result.keyword_coverage:.1f}% "
                f"({retrieval_result.keywords_found}/{retrieval_result.total_keywords})"
            )
        except Exception:
            logger.exception("Retrieval evaluation failed for test %s", index)
            print(f"ERROR: Retrieval evaluation failed for test #{index}. Skipping.")
            print()
            continue

        if not retrieval_only:
            try:
                answer_result, generated_answer, _ = evaluate_answer(test)
                answer_scores.append(answer_result)
                print(f"Generated Answer: {generated_answer}")
                print(
                    f"Answer — Accuracy: {answer_result.accuracy:.2f}/5 | "
                    f"Completeness: {answer_result.completeness:.2f}/5 | "
                    f"Relevance: {answer_result.relevance:.2f}/5"
                )
                print(f"Feedback: {answer_result.feedback}")
            except Exception:
                logger.exception("Answer evaluation failed for test %s", index)
                print(f"ERROR: Answer evaluation failed for test #{index}. Skipping.")
        print()

    completed = len(retrieval_scores)
    print(f"{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Tests completed: {completed}/{total}")
    if completed:
        print(f"Avg MRR: {sum(r.mrr for r in retrieval_scores) / total:.4f}")
        print(f"Avg nDCG: {sum(r.ndcg for r in retrieval_scores) / total:.4f}")
        print(
            f"Avg Keyword Coverage: "
            f"{sum(r.keyword_coverage for r in retrieval_scores) / total:.1f}%"
        )

    if answer_scores:
        print(f"Avg Accuracy: {sum(a.accuracy for a in answer_scores) / total:.2f}/5")
        print(
            f"Avg Completeness: {sum(a.completeness for a in answer_scores) / total:.2f}/5"
        )
        print(f"Avg Relevance: {sum(a.relevance for a in answer_scores) / total:.2f}/5")
    print(f"{'=' * 80}\n")


def main():
    """CLI to evaluate one test, or all tests."""

    usage = (
        "Usage:\n"
        "  uv run python -m src.evaluation.eval <test_row_number>\n"
        "  uv run python -m src.evaluation.eval all\n"
        "  uv run python -m src.evaluation.eval all --retrieval-only\n"
        "  uv run python -m src.evaluation.eval all --limit 10\n"
        "  uv run python -m src.evaluation.eval all --retrieval-only --limit 10"
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == "all":
        retrieval_only = "--retrieval-only" in sys.argv
        limit = None
        if "--limit" in sys.argv:
            try:
                limit_index = sys.argv.index("--limit")
                limit = int(sys.argv[limit_index + 1])
                if limit < 1:
                    raise ValueError
            except (IndexError, ValueError):
                print("Error: --limit requires a positive integer")
                sys.exit(1)
        run_all_evaluations(limit=limit, retrieval_only=retrieval_only)
        return

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer, or use 'all'")
        print(usage)
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
