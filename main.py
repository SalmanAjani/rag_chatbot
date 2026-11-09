# from src.llms.llm import llm
# from src.prompts.prompt import prompt_template
# from src.retrieval.retriever import question, retrieved_docs

# context = "\n\n".join(document.page_content for document in retrieved_docs)


# formatted_prompt = prompt_template.format(context=context, question=question)

# response = llm.invoke(formatted_prompt)

# print("Answer is:", response.content)

from fastapi import FastAPI

from src.api.chat_router import router

app = FastAPI()
app.include_router(router)
