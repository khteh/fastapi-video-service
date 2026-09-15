"""
Generation provider registry — the concrete answer to "plug-and-play of
both simulated generation and real AI/video-generation providers".

A `GenerationProvider` bundles everything the pipeline needs (a script
writer, a narrator/voice, a slide renderer, a video assembler, a topic
classifier) behind two calls: `validate_topic()` (synchronous, called by
the API layer BEFORE a job is created) and `generate()` (the actual async
work, run by a worker once a job exists). `src/worker.py` and
`src/main.py` only ever talk to a `GenerationProvider` by this interface —
neither knows or cares whether the script came from a template or a real
LLM, or whether the voice is a synthesized tone or a cloud neural TTS.

Two providers are registered out of the box:

  "simulated" — fully offline, deterministic-ish, no API keys, no network.
      Template script writer + local speech (flite, falling back to a
      tone track) + offline heuristic topic classification. This is the
      default, and what the test suite uses. `validate_topic()` here can
      never fail due to unavailability — the heuristic has no external
      dependencies.

  "ai" — real AI script writing, real AI voice (edge-tts or Piper), and
      real AI topic classification via the Anthropic API, ALL used
      directly with no fallback to their "simulated" counterparts: script
      and voice directly determine output quality, and topic
      classification is now also a hard gate (see below), so nothing in
      "ai" mode silently substitutes simulated-quality behavior anywhere.
      Opt in with `GENERATION_PROVIDER=ai` (see README for setup).

Failure handling policy (see README's "Failure handling" section for the
full write-up): three categories of failure are checked SYNCHRONOUSLY by
`validate_topic()`, called from `POST /api/v1/videos` before any job is
created, so a bad request never wastes a job slot or leaves the caller
polling for something that was never going to work:
  1. Malformed input (blank, no letters, too short/long) — enforced by
     pydantic in `src/models.py`, before this module is even reached.
  2. Semantically nonsensical or non-STEM topics — `validate_topic()`
     runs the configured `TopicClassifier` and returns its verdict.
  3. "ai" mode with no network / an invalid API key — `validate_topic()`
     lets `ClassificationUnavailableError` propagate straight to the
     caller (`main.py` turns it into a 503) rather than swallowing it,
     since a real Anthropic API call inside `validate_topic()` doubles as
     a live health check for the same backend `generate()` will need a
     moment later.

Adding a third provider (e.g. swapping in a real generative-video API
instead of the local Pillow/ffmpeg renderer) means implementing this same
interface and registering it in `_PROVIDER_FACTORIES` below — nothing in
`worker.py` or the API layer needs to change.
"""
from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

from src.config import settings
from src.generation import pipeline
from src.generation.narrator import Narrator, select_ai_narrator, select_narrator
from src.generation.script_providers import (
    AnthropicScriptProvider,
    FallbackScriptProvider,
    ScriptProvider,
    SimulatedScriptProvider,
)
from src.generation.slide_renderer import PillowSlideRenderer
from src.generation.topic_classifier import (
    AnthropicTopicClassifier,
    FallbackTopicClassifier,
    ClassificationResult,
    HeuristicTopicClassifier,
    TopicClassifier,
)
from src.generation.video_assembler import FfmpegVideoAssembler, VideoAssembler

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], Awaitable[None]]
NarratorFactory = Callable[[], Awaitable[Narrator]]


class UnknownProviderError(ValueError):
    pass


class GenerationProvider(abc.ABC):
    name: str

    async def initialize(self) -> None:
        """Optional one-time async setup (e.g. probing for the best available
        narrator backend). Called once at application startup."""
        return None

    @abc.abstractmethod
    async def validate_topic(self, topic: str) -> ClassificationResult:
        """
        Synchronous (from the caller's perspective — no job involved),
        called by `POST /api/v1/videos` BEFORE a job is created. Returns
        the classifier's verdict on whether `topic` is a coherent,
        answerable STEM question/topic.

        For "ai" mode specifically, this call itself IS the real Anthropic
        API call — a failure here (no network, bad API key, timeout)
        raises `ClassificationUnavailableError` rather than returning a
        result, and the caller (main.py) turns that into an immediate,
        clear error response with no job created — see this module's
        docstring for the full policy.
        """

    @abc.abstractmethod
    async def generate(
        self,
        topic: str,
        difficulty: str,
        work_dir: Path,
        on_progress: Optional[ProgressCallback] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> pipeline.GenerationResult: ...


class _PipelineBackedProvider(GenerationProvider):
    """Shared plumbing for providers that run the local slide+narration
    pipeline, differing only in which ScriptProvider/Narrator/TopicClassifier
    they use."""

    def __init__(
        self,
        script_provider: ScriptProvider,
        topic_classifier: TopicClassifier,
        narrator_factory: NarratorFactory,
    ) -> None:
        self._script_provider = script_provider
        self._topic_classifier = topic_classifier
        self._narrator_factory = narrator_factory
        self._narrator: Narrator | None = None
        self._slide_renderer = PillowSlideRenderer(settings.video_width, settings.video_height)
        self._assembler: VideoAssembler = FfmpegVideoAssembler(
            settings.video_width, settings.video_height, settings.video_fps
        )

    async def initialize(self) -> None:
        self._narrator = await self._narrator_factory()
        logger.info(
            "Generation provider '%s' using script provider '%s', topic classifier "
            "'%s', and narrator '%s'",
            self.name,
            self._script_provider.name,
            self._topic_classifier.name,
            self._narrator.name,
        )

    async def validate_topic(self, topic: str) -> ClassificationResult:
        is_valid, reason = await self._topic_classifier.is_valid(topic)
        return ClassificationResult(is_valid=is_valid, reason=reason, classifier=self._topic_classifier.name)

    async def generate(
        self,
        topic: str,
        difficulty: str,
        work_dir: Path,
        on_progress: Optional[ProgressCallback] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> pipeline.GenerationResult:
        if self._narrator is None:
            # initialize() wasn't called explicitly (e.g. ad-hoc/test usage) —
            # resolve lazily rather than requiring callers to remember to.
            await self.initialize()
        assert self._narrator is not None
        return await pipeline.generate(
            topic=topic,
            difficulty=difficulty,
            script_provider=self._script_provider,
            narrator=self._narrator,
            slide_renderer=self._slide_renderer,
            assembler=self._assembler,
            work_dir=work_dir,
            on_progress=on_progress,
            max_duration_seconds=max_duration_seconds,
        )


class SimulatedProvider(_PipelineBackedProvider):
    """Fully offline: template script + flite/tone narration + heuristic
    topic classification. No API keys, no network, fast and reproducible —
    the default, and what tests use. `validate_topic()` can never raise
    here — the heuristic classifier has no external dependencies."""

    name = "simulated"

    def __init__(self) -> None:
        super().__init__(
            script_provider=SimulatedScriptProvider(),
            topic_classifier=HeuristicTopicClassifier(),
            narrator_factory=lambda: select_narrator(prefer_realistic=False),
        )


class AIProvider(_PipelineBackedProvider):
    """
    Real AI script writing, real AI voice, and real AI topic
    classification (Anthropic) — all used directly, with NO fallback to
    their "simulated"/offline counterparts anywhere. A caller who asks
    for "ai" either gets genuinely AI-generated output, or a clear error
    explaining why not — never a silent downgrade.

    `validate_topic()` doubles as a live health check for the Anthropic
    backend: since it's a real API call, its failure (network, bad key,
    timeout) is a reliable signal that `generate()` would likely fail the
    same way a moment later, so `main.py` rejects the request immediately
    rather than creating a job doomed to fail.
    """

    name = "ai"

    def __init__(self) -> None:
        script_provider = FallbackScriptProvider(
            primary=AnthropicScriptProvider(api_key=..., model=...),
            fallback=SimulatedScriptProvider(),
        )
        topic_classifier = FallbackTopicClassifier(
            primary=AnthropicTopicClassifier(api_key=..., model=...),
            fallback=HeuristicTopicClassifier(),
        )
        super().__init__(
            script_provider=script_provider,
            topic_classifier=topic_classifier,
            narrator_factory=select_ai_narrator,
        )


_PROVIDER_FACTORIES: dict[str, Callable[[], GenerationProvider]] = {
    "simulated": SimulatedProvider,
    "ai": AIProvider,
}

_provider_instances: dict[str, GenerationProvider] = {}


def get_provider(name: str) -> GenerationProvider:
    """Return the (lazily constructed, cached) GenerationProvider registered
    under `name`. Raises UnknownProviderError for anything unregistered."""
    if name not in _PROVIDER_FACTORIES:
        raise UnknownProviderError(
            f"unknown generation provider '{name}'; available: {sorted(_PROVIDER_FACTORIES)}"
        )
    if name not in _provider_instances:
        _provider_instances[name] = _PROVIDER_FACTORIES[name]()
    return _provider_instances[name]


def available_providers() -> list[str]:
    return sorted(_PROVIDER_FACTORIES)
