import gradio as gr

from src.chat.rag import ask_question


def respond(message: str, history: list) -> str:
    return ask_question(message)


demo = gr.ChatInterface(
    fn=respond,
    title="RAG Chatbot",
    description="Ask a question about the company",
    examples=["What is the best car?", "What is 3C?", "What is your privacy policy?"],
)
