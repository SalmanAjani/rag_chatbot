from langchain_openai import OpenAIEmbeddings

from src.core.config import config

embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    api_key=config.OPENROUTER_API_KEY,
    base_url=config.OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
)
