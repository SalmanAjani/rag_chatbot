import logging

from langchain_chroma import Chroma

from src.core.config import config
from src.ingestion.embedding import embeddings
from src.ingestion.splitter import chunks

logger = logging.getLogger(__name__)


def _build_vector_store() -> Chroma:
    store = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    )

    if store._collection.count() == len(chunks):
        return store

    if store._collection.count() > 0:
        store.delete_collection()

    return Chroma.from_documents(
        collection_name=config.CHROMA_COLLECTION_NAME,
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    )


try:
    vector_store = _build_vector_store()
except Exception:
    logger.exception("Failed to initialize the vector store")
    raise RuntimeError(
        "Could not start the knowledge base. Check your embedding setup and try again."
    ) from None
