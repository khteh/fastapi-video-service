"""
Background worker pool: the only place that talks to *all* layers (state,
persistence-via-state, artifacts, and generation) at once, to orchestrate
turning a queued request into a stored, retrievable video.

The API layer never touches generation/artifacts directly, and the
generation layer never touches state/artifacts directly — this module is
the seam that wires them together, which is what keeps each layer
independently testable. It's also deliberately agnostic to *which*
generation provider is in play: it asks `providers.get_provider(name)` for
whatever's configured and calls its `generate()` method — swapping
"simulated" for "ai" (or a future provider) requires no changes here.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from src.artifacts.store import ArtifactStore
from src.generation.narrator import NarrationUnavailableError
from src.generation.pipeline import GenerationError
from src.generation.providers import GenerationProvider, get_provider
from src.generation.script_providers import ProviderUnavailableError
from src.generation.topic_classifier import ClassificationUnavailableError
from src.models import JobStatus, VideoRequest
from src.state.job_state import InvalidTransitionError, JobStateManager

logger = logging.getLogger(__name__)

class VideoWorkerPool:
    def __init__(
        self,
        state: JobStateManager,
        artifacts: ArtifactStore,
        num_workers: int,
        provider_name: str,
    ) -> None:
        self._state = state
        self._artifacts = artifacts
        self._num_workers = num_workers
        self._provider_name = provider_name
        self._provider: GenerationProvider | None = None
        self._queue: "asyncio.Queue[tuple[str, VideoRequest]]" = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._job_tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        self._provider = get_provider(self._provider_name)
        await self._provider.initialize()
        self._workers = [
            asyncio.create_task(self._worker_loop(i)) for i in range(self._num_workers)
        ]
        logger.info(f"Started {self._num_workers} video generation workers using provider '{self._provider_name}'")

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        for task in self._job_tasks.values():
            task.cancel()
        self._workers.clear()

    async def enqueue(self, job_id: str, request: VideoRequest) -> None:
        await self._queue.put((job_id, request))

    def get_running_task(self, job_id: str) -> asyncio.Task | None:
        return self._job_tasks.get(job_id)

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            job_id, request = await self._queue.get()
            try:
                current = await self._state.get(job_id)
            except KeyError as ke:
                logger.exception("Job %s not found: %s", job_id, ke)
                self._queue.task_done()
                continue
            if current.status == JobStatus.CANCELLED:
                logger.debug("Skipping job %s: cancelled while queued", job_id)
                self._queue.task_done()
                continue

            logger.debug("Worker %d picked up job %s", worker_index, job_id)
            job_task = asyncio.create_task(self._process_job(job_id, request))
            self._job_tasks[job_id] = job_task
            try:
                await job_task
            except asyncio.CancelledError:
                # This fires in two distinct situations that must be told
                # apart: (a) job_task was cancelled directly (e.g. DELETE
                # /api/v1/videos/{job_id}) - the worker loop itself is
                # fine and should continue to the next queued job; (b)
                # this worker loop's own task was cancelled (e.g.
                # VideoWorkerPool.stop() during shutdown) - cancelling an
                # outer task that's awaiting an inner one propagates into
                # the inner task too, so it lands here looking identical.
                # Swallowing it in case (b) would leave this loop calling
                # `await self._queue.get()` forever on an empty queue,
                # hanging stop()'s `await asyncio.gather(*self._workers)`
                # indefinitely. Task.cancelling() (3.11+) tells them apart.
                if asyncio.current_task().cancelling() > 0:
                    raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("Worker %d crashed processing job %s", worker_index, job_id)
            finally:
                self._job_tasks.pop(job_id, None)
                self._queue.task_done()

    async def _process_job(self, job_id: str, request: VideoRequest) -> None:
        assert self._provider is not None, "worker pool not started"

        async def on_progress(progress: int, stage: str) -> None:
            try:
                await self._state.update_progress(job_id, stage=stage, progress=progress)
            except KeyError as ke:
                logger.exception("Job %s not found: %s", job_id, ke)

        work_dir = Path(tempfile.mkdtemp(prefix=f"stemvideo_{job_id}_"))
        try:
            await self._state.mark_processing(job_id, stage="starting", progress=0)

            result = await self._provider.generate(
                topic=request.topic,
                difficulty=request.difficulty.value,
                work_dir=work_dir,
                on_progress=on_progress,
            )

            artifact = await self._artifacts.save(job_id, result.video_path, result.duration_seconds)
            await self._state.mark_completed(job_id, artifact)
            logger.info("Job %s completed: %s (%.1fs)", job_id, artifact, result.duration_seconds)

        except asyncio.CancelledError:
            try:
                await self._state.mark_cancelled(job_id)
            except (KeyError, InvalidTransitionError) as e:
                logger.exception("Job %s not found or invalid transition: %s", job_id, e)
            raise

        except (
            GenerationError,
            ProviderUnavailableError,
            NarrationUnavailableError,
            # Topic classification now runs synchronously in main.py before
            # a job is ever created (see providers.py's validate_topic()),
            # so this shouldn't fire in the normal path — kept as a
            # defensive catch in case that invariant is ever broken.
            ClassificationUnavailableError,
        ) as exc:
            logger.warning("Job %s failed: %s", job_id, exc)
            await self._safe_mark_failed(job_id, str(exc))

        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Job %s crashed unexpectedly", job_id)
            await self._safe_mark_failed(job_id, f"internal error: {exc}")

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _safe_mark_failed(self, job_id: str, error: str) -> None:
        try:
            await self._state.mark_failed(job_id, error)
        except KeyError as ke:
            logger.exception("Job %s not found: %s", job_id, ke)
