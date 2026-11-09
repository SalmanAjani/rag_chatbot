from langchain_community.document_loaders import DirectoryLoader, TextLoader

from src.core.config import config

loader = DirectoryLoader(
    config.DATA_DIR,
    glob=config.DATA_GLOB,
    loader_cls=TextLoader,
)

documents = loader.load()
