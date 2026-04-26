from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM — Ollama local (Qwen3-4B) or Groq API fallback
    llm_backend: str = "ollama"          # "ollama" | "groq"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"

    # Embeddings — FastEmbed ONNX (no Ollama/torch, multilingual, handles Hindi)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Vector store
    chroma_persist_dir: str = "./data/chroma"

    # Twilio WhatsApp
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = "whatsapp:+14155238886"  # sandbox default

    # RAG
    retrieval_k: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Hindi system prompt
    system_prompt: str = (
        "आप एक सहायक AI हैं जो हिंदी में उत्तर देते हैं। "
        "दिए गए संदर्भ के आधार पर सटीक और सरल उत्तर दें। "
        "यदि उत्तर संदर्भ में नहीं है, तो कहें 'मुझे इस बारे में जानकारी नहीं है।'"
    )

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
