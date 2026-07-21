from fastapi import FastAPI
from app.api import routes_documents, routes_chat, routes_health, routes_evaluation
from app.config import settings
from app.db.session import engine
from app.db.models import Base
from app.logging_config import get_logger
from app.vectorstore.pinecone_client import ensure_index_exists

logger = get_logger(__name__)

app = FastAPI(title="Production RAG Chatbot", version="1.0.0")

@app.on_event("startup")
def on_startup():
    logger.info("Starting up: ensuring DB tables and Pinecone index exist")
    Base.metadata.create_all(bind=engine)  # dev convenience; use Alembic migrations in real prod
    ensure_index_exists()
    if settings.LANGSMITH_ENABLED:
        from app.evaluation.langsmith_evaluator import configure_tracing
        configure_tracing()


app.include_router(routes_documents.router)
app.include_router(routes_chat.router)
app.include_router(routes_health.router)
app.include_router(routes_evaluation.router)
