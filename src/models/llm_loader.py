# models/llm_loader.py
from langchain_community.chat_models import ChatOllama

def load_llama_model(model_name: str = "llama3.2", base_url: str = "http://localhost:11434", temperature: float = 0.4):
    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
