import logging

from src.llms.llm import llm
from src.prompts.prompt import prompt_template
from src.retrieval.retriever import retriever

logger = logging.getLogger(__name__)


class ServiceUnavailableError(Exception):
    """Raised when retrieval or the LLM temporarily fails"""


def fetch_context(question: str, filter: dict | None = None) -> list:
    """
    Retrieve documents for a question (used by the app and by evaluation).

    filter is optional metadata, e.g. {"source_file": "faq.txt"}.
    Evaluation and existing callers keep working with fetch_context(question).
    """
    return retriever.invoke(question, filter=filter)


def _format_context(documents: list) -> str:
    blocks = []
    for i, document in enumerate(documents, start=1):
        blocks.append(f"[Excerpt {i}]\n{document.page_content}")

    return "\n\n".join(blocks)


def answer_question(question: str) -> tuple[str, list]:
    """
    Run the full RAG pipeline.
    Returns the generated answer and the retrieved documents.
    """

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    try:
        retrieved_docs = fetch_context(question)
    except Exception:
        logger.exception("Retrieval Failed")

        raise ServiceUnavailableError(
            "Sorry, I couldn't search the knowledge base right now. Please try again later"
        )

    context = _format_context(retrieved_docs)

    formatted_prompt = prompt_template.format(
        context=context,
        question=question,
    )

    try:
        response = llm.invoke(formatted_prompt)
    except Exception:
        logger.exception("LLM request failed")
        raise ServiceUnavailableError(
            "Sorry, the AI service is temporarily unavailable. Please try again later."
        )

    return response.content, retrieved_docs


def ask_question(question: str) -> str:
    """Return only the answer string (used by the UI and API)."""
    answer, _ = answer_question(question)
    return answer
