from fastapi import APIRouter, Response

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live")
async def live():
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    """Readiness probe."""
    from app.main import app_state

    if app_state["models_ready"]:
        return {"status": "ok"}
    return Response(
        content='{"status": "not_ready"}',
        status_code=503,
        media_type="application/json",
    )
