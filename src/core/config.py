import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENROUTER_API_KEY = os.getenv("API_KEY")
    OPENROUTER_MODEL_NAME = os.getenv("MODEL")
    OPENROUTER_BASE_URL = os.getenv("BASE_URL")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME = os.getenv("GROQ_MODEL")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # Document loading
    DATA_DIR = "data/"
    DATA_GLOB = "**/*.txt"

    # Text splitting
    CHUNK_SIZE = 700
    CHUNK_OVERLAP = 120

    # Vector store
    CHROMA_COLLECTION_NAME = "chat_docs"
    CHROMA_PERSIST_DIRECTORY = "data/chroma_db"

    # Retrieval
    # SEARCH_TYPE = "hybrid"
    SEARCH_TYPE = "similarity"
    RETRIEVER_K = 3

    BM25_K = 20
    DENSE_K = 20

    RRF_K = 60
    FUSION_LIMIT = 40

    RERANK_TOP_K = 5
    RERANK_ENABLED = True
    RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
    RERANK_CACHE_DIR = "data/flashrank_cache"


config = Config()
