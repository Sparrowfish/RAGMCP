# src/api/routes/ingest.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger
from src.api.schemas import IngestRequest, IngestResponse
from src.api.dependencies import state
from src.rag.pipeline import ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ingestion_running = False


@router.post("", response_model=IngestResponse)
async def run_ingestion(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Trigger document ingestion. Runs in the background so the API stays responsive.
    After ingestion the vector store is hot-swapped into state without restart.
    """
    global _ingestion_running

    if _ingestion_running:
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    def _ingest_task():
        global _ingestion_running
        _ingestion_running = True
        try:
            logger.info(f"Background ingestion started: {request.input_dir}")
            vs = ingest(request.input_dir, state.cfg)
            state.vectorstore = vs  # hot-swap — no restart needed
            logger.success(f"Ingestion complete. {vs.index.ntotal} vectors loaded into memory.")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
        finally:
            _ingestion_running = False

    background_tasks.add_task(_ingest_task)

    return IngestResponse(
        status="started",
        vectors_stored=0,
        message=f"Ingestion started for '{request.input_dir}'. Poll /health to confirm when ready.",
    )