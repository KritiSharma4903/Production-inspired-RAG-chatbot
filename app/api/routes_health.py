"""
routes_health.py
=================
WHY THIS FILE EXISTS:
Load balancers, container orchestrators (k8s liveness/readiness probes),
and uptime monitors all need a cheap endpoint to hit. `/health/live` checks
the process is up; `/health/ready` additionally checks the DB and vector DB
are reachable -- so a pod that's running but can't reach Postgres gets
pulled out of rotation instead of serving broken requests.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.vectorstore.pinecone_client import get_pinecone_client
from app.config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def liveness():
    return {"status": "alive"}


@router.get("/ready")
def readiness(db: Session = Depends(get_db)):
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    try:
        pc = get_pinecone_client()
        pc.list_indexes()
        checks["pinecone"] = "ok"
    except Exception as e:
        checks["pinecone"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
