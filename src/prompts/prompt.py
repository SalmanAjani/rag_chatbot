from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(
    """
    You are an AI assistant.
    If you don't know the answer in the context, say "I don't know".
    Context: {context} Question: {question}
    Answer:
    """
)
