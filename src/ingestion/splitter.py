from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import config
from src.ingestion.ingest import documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n---", "\n\n", "\nQ: ", "\n", " ", ""],
)

chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    source = chunk.metadata.get("source", "")

    # "data/faq.txt" -> split it -> ["data", "faq.txt"]
    file_name = source.split("/")[-1] if source else "unknown"

    chunk.page_content = f"Source: {file_name}\n{chunk.page_content}"
    """
    Source: faq.txt
    How to reset your password?
    Go to settings and click reset password.
    """

    chunk.metadata["source_file"] = file_name
    """
    {
        "source": "data/faq.txt",
        "source_file": "faq.txt"
    }   
    """
