from langchain_openrouter import ChatOpenRouter

# from langchain_groq import ChatGroq
from src.core.config import config

llm = ChatOpenRouter(
    model=config.OPENROUTER_MODEL_NAME,
    api_key=config.OPENROUTER_API_KEY,
    temperature=0.2,
)
# llm = ChatGroq(
#     model=config.GROQ_MODEL_NAME,
#     api_key=config.GROQ_API_KEY,
#     base_url=config.GROQ_BASE_URL,
#     temperature=0.2,
# )
