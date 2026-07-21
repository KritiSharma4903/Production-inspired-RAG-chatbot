from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    DATABASE_URL: str

    # --- Pinecone ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-chatbot-index"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_NAMESPACE: str = "default"
    EMBEDDING_DIMENSION: int = 384

    # --- Groq ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-70b-versatile"

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- File storage ---
    FILE_STORAGE_ROOT: str = "./data/raw_files"

    # --- App behavior ---
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    CHUNK_TOKEN_SIZE: int = 400
    CHUNK_OVERLAP_TOKENS: int = 60
    TOP_K_RETRIEVAL: int = 5

    RAGAS_ENABLED: bool = True
    DEEPEVAL_ENABLED: bool = False
    LANGSMITH_ENABLED: bool = False

    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "Production-RAG"
    
    GOOGLE_API_KEY: str | None = None

    EVAL_JUDGE_MODEL: str = "llama-3.3-70b-versatile"  # Groq model used by all native/adapter LLM-judge calls


# Instantiated ONCE at import time. Every other module does:
#   from app.config import settings
# instead of re-reading environment variables. This also means a missing
# required variable raises a validation error immediately on process start.
settings = Settings()
