import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()


llm = ChatOpenRouter(model=os.getenv("MODEL"), api_key=os.getenv("API_KEY"))
