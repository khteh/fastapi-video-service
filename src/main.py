"""
STEM Concept Explanation Video Service — API layer.

Responsibility: HTTP request/response handling only. This module wires
together the state, persistence, artifact, and worker layers at startup
(see `lifespan`) but contains no business logic itself — job lifecycle
rules live in `src/state`, durability in `src/persistence`, video file
handling in `src/artifacts`, and the actual generation in `src/generation`
+ `src/worker` (itself pluggable between a "simulated" and an "ai"
generation provider — see `src/generation/providers.py`).

Endpoints:
  POST   /api/v1/videos            submit a new video generation request
  GET    /api/v1/videos            list all requested videos/jobs
  GET    /api/v1/videos/{job_id}   status of one job
  GET    /api/v1/videos/{job_id}/download   open/retrieve the finished video
  DELETE /api/v1/videos/{job_id}   cancel a pending/in-progress job
  GET    /api/v1/providers         list available generation providers
  GET    /health                   liveness probe
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse

from src.artifacts.store import ArtifactNotFoundError, LocalArtifactStore
from src.config import settings
from src.generation.providers import available_providers, get_provider
from src.generation.topic_classifier import ClassificationUnavailableError
from src.models import JobRecord, JobStatus, JobSubmittedResponse, VideoRequest
from src.persistence.repository import JobNotFoundError, JsonFileJobRepository
from src.state.job_state import InvalidTransitionError, JobStateManager
from src.worker import VideoWorkerPool


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = JsonFileJobRepository(settings.jobs_dir)
    await repository.load_from_disk()  # recover job history across restarts

    state = JobStateManager(repository)
    artifacts = LocalArtifactStore(settings.artifacts_dir)
    worker_pool = VideoWorkerPool(
        state=state,
        artifacts=artifacts,
        num_workers=settings.num_workers,
        provider_name=settings.generation_provider,
    )

    # Jobs left PENDING/PROCESSING from a previous run were interrupted by
    # a crash/restart — nothing resumes them, so mark them FAILED now
    # rather than leaving them stuck forever.
    recovered = await state.recover_interrupted_jobs()
    if recovered:
        logging.getLogger(__name__).warning(
            "Marked %d interrupted job(s) as failed on startup", len(recovered)
        )

    await worker_pool.start()

    app.state.job_state = state
    app.state.artifacts = artifacts
    app.state.worker_pool = worker_pool

    try:
        yield
    finally:
        await worker_pool.stop()

app = FastAPI(
    title="STEM Concept Video Service",
    description=(
        "Request short explainer videos (visuals + narrated audio) for "
        "Science, Technology, Engineering, and Math topics, generated "
        "asynchronously. Videos are capped at "
        f"{settings.max_duration_seconds:.0f} seconds. Generation backend is "
        f"pluggable — currently running the '{settings.generation_provider}' provider."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _state(request: Request) -> JobStateManager:
    return request.app.state.job_state


def _artifacts(request: Request) -> LocalArtifactStore:
    return request.app.state.artifacts


def _worker_pool(request: Request) -> VideoWorkerPool:
    return request.app.state.worker_pool


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/providers", tags=["ops"])
async def list_providers() -> dict:
    """List registered generation providers and which one is active."""
    return {"active": settings.generation_provider, "available": available_providers()}


@app.post(
    "/api/v1/videos",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["videos"],
)
async def request_video(payload: VideoRequest, request: Request) -> JobSubmittedResponse:
    """Submit a new STEM explainer video request.

    `topic` can be any STEM question or concept a learner is curious about,
    e.g. "How does the pH scale work?", "Why do atoms form covalent
    bonds?", or "What is the difference between ionic and covalent
    bonding?". Returns immediately with a job id; the video (visuals +
    narrated audio, up to 90 seconds) is generated asynchronously in the
    background. Poll GET /api/v1/videos/{job_id} for status, then GET
    .../download once status is "completed".

    Three categories of failure are rejected here, synchronously, with NO
    job created — see the README's "Failure handling" section for the
    full policy:
      1. Malformed input (blank, no letters, wrong length) — pydantic.
      2. Semantically nonsensical or non-STEM topics — checked against
         the configured topic classifier.
      3. In "ai" mode, no network or an invalid API key — the same
         classifier call above doubles as a live health check for the
         Anthropic backend; its failure is reported immediately as a 503
         rather than discovered later via a doomed job.
    """
    if len(payload.topic) < settings.min_topic_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"topic must be at least {settings.min_topic_length} characters",
        )

    provider = get_provider(settings.generation_provider)
    try:
        classification = await provider.validate_topic(payload.topic)
    except ClassificationUnavailableError as exc:
        logging.warning(f"Topic classification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"AI generation ('{settings.generation_provider}' provider) is "
                f"currently unavailable: {exc}. Check ANTHROPIC_API_KEY and network "
                f"connectivity, or submit with a provider that doesn't require them."
            ),
        ) from exc

    if not classification.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"topic rejected: {classification.reason}",
        )

    state = _state(request)
    worker_pool = _worker_pool(request)

    job = await state.create(payload, provider=settings.generation_provider)
    await worker_pool.enqueue(job.job_id, payload)

    return JobSubmittedResponse(
        job_id=job.job_id,
        status=job.status,
        provider=job.provider,
        poll_url=f"/api/v1/videos/{job.job_id}",
    )


@app.get("/api/v1/videos", response_model=list[JobRecord], tags=["videos"])
async def list_videos(request: Request) -> list[JobRecord]:
    """List all requested videos/jobs, most recent first, with each job's
    current status visible (pending/processing/completed/failed/cancelled)."""
    return await _state(request).list_all()


@app.get("/api/v1/videos/{job_id}", response_model=JobRecord, tags=["videos"])
async def get_video_status(job_id: str, request: Request) -> JobRecord:
    """Get the current status (and result metadata, once completed) of a video job."""
    try:
        return await _state(request).get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found"
        ) from exc


@app.get("/api/v1/videos/{job_id}/download", tags=["videos"])
async def download_video(job_id: str, request: Request) -> FileResponse:
    """Retrieve/open the finished video artifact for a completed job.

    Streams the actual .mp4 file with a video/mp4 content type, so it can
    be opened directly in a browser tab or downloaded.
    """
    state = _state(request)
    artifacts = _artifacts(request)

    try:
        job = await state.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found"
        ) from exc

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job '{job_id}' is not completed yet (status: {job.status.value})",
        )

    try:
        path = await artifacts.path_for(job_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"video artifact for job '{job_id}' was not found in storage",
        ) from exc

    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@app.delete("/api/v1/videos/{job_id}", response_model=JobRecord, tags=["videos"])
async def cancel_video(job_id: str, request: Request) -> JobRecord:
    """Cancel a pending or in-progress job."""
    state = _state(request)
    worker_pool = _worker_pool(request)

    try:
        job = await state.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found"
        ) from exc

    if job.status.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job '{job_id}' is already {job.status.value} and cannot be cancelled",
        )

    task = worker_pool.get_running_task(job_id)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await state.get(job_id)

    # Not yet picked up by a worker: mark cancelled directly. The worker
    # loop checks for this status before processing a dequeued job, so it
    # won't be silently overwritten later.
    try:
        return await state.mark_cancelled(job_id)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
