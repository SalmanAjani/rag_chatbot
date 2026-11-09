from src.core.config import config
from src.ingestion.vector_store import vector_store

retriever = vector_store.as_retriever(
    search_type=config.SEARCH_TYPE, search_kwargs={"k": config.RETRIEVER_K}
)
