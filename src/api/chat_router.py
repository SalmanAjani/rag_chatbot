from fastapi import APIRouter

from src.api.chat_schema import ChatRequest, ChatResponse
from src.llms.llm import llm
from src.prompts.prompt import prompt_template
from src.retrieval.retriever import retriever

router = APIRouter()


@router.post("/ask")
def ask_question(request: ChatRequest):
    retrieved_docs = retriever.invoke(request.question)
    context = "\n\n".join(document.page_content for document in retrieved_docs)

    formatted_prompt = prompt_template.format(
        context=context, question=request.question
    )

    response = llm.invoke(formatted_prompt)

    return ChatResponse(answer=response.content)
