"""
Narration (text-to-speech) for the video generation pipeline.

Four implementations, in increasing order of voice realism:

- `ToneNarrator`: dependency-free fallback — a paced tone track (duration
  estimated from word count) so the pipeline always produces *a* correctly
  timed audio track even with nothing else available.
- `FliteNarrator`: real but robotic offline speech via ffmpeg's built-in
  `flite` lavfi source (classic diphone synthesis). No extra Python deps,
  just an ffmpeg build with flite support.
- `PiperNarrator`: natural-sounding **offline neural** TTS via the Piper
  engine (`uv sync --extra ai`, plus a one-time voice model download —
  see README). Genuinely authentic-sounding, fully local after setup.
- `EdgeTTSNarrator`: natural-sounding **cloud neural** TTS via Microsoft
  Edge's free online voices (`uv sync --extra ai`). The most realistic
  option, but needs network access at synthesis time.

`select_narrator(prefer_realistic=...)` builds a `FallbackNarrator` chain
covering every backend that passes its (best-effort) `is_available()`
check at startup, in priority order, always ending in `ToneNarrator`.

Critically, this fallback is re-checked on **every synthesis call**, not
just once at startup: `is_available()` for a network-backed narrator like
`EdgeTTSNarrator` can only check "is the package installed", not "is the
network actually reachable right now" — an installed-but-unreachable
backend would pass that check yet still fail the moment it tries a real
request. Each backend below normalizes any synthesis-time failure
(network error, subprocess failure, malformed output) into
`NarrationUnavailableError`, and `FallbackNarrator` catches that on every
call to cascade to the next backend — so a network outage during
narration degrades gracefully and *recovers automatically* once the
network returns, rather than sticking with a single bad choice for the
rest of the process's lifetime.
"""
from __future__ import annotations

import abc, wave, logging
from pathlib import Path

import numpy as np

from src.config import settings
from src.generation.subprocess_utils import SubprocessError, probe_duration, run_subprocess

_WORDS_PER_MINUTE = 145  # rough average narration pace


class NarrationUnavailableError(Exception):
    """Raised when a narrator backend fails at synthesis time in a way
    that should trigger falling back to the next backend in the chain
    (network error, subprocess failure, malformed output) — as opposed to
    a genuinely unexpected bug, which is left to propagate normally."""


class Narrator(abc.ABC):
    name: str

    @abc.abstractmethod
    async def synthesize(self, text: str, out_path: Path) -> float:
        """Write narration audio for `text` to `out_path` (WAV). Returns duration in seconds.

        Implementations should raise `NarrationUnavailableError` (not let
        a raw network/subprocess exception propagate) for failures that a
        different backend might recover from, so a `FallbackNarrator`
        wrapping this one can catch it and cascade."""

    async def is_available(self) -> bool:
        """Cheap-ish check for whether this backend can actually run right now."""
        return True


class ToneNarrator(Narrator):
    """Fallback: a soft paced tone track, timed to roughly match spoken narration length."""

    name = "tone"

    def __init__(self, sample_rate: int = 22050) -> None:
        self._sample_rate = sample_rate

    def _estimate_duration(self, text: str) -> float:
        word_count = max(1, len(text.split()))
        minutes = word_count / _WORDS_PER_MINUTE
        return max(1.5, minutes * 60.0)

    async def synthesize(self, text: str, out_path: Path) -> float:
        duration = self._estimate_duration(text)
        sr = self._sample_rate
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # A gentle two-tone "narration cue" envelope rather than a harsh
        # constant beep: gives each slide an audible, distinct audio track
        # without pretending to be real speech.
        base_freq = 220.0
        wave_data = 0.15 * np.sin(2 * np.pi * base_freq * t)
        fade_len = int(sr * 0.05)
        envelope = np.ones_like(wave_data)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        samples = (wave_data * envelope * 32767).astype(np.int16)

        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(samples.tobytes())

        return duration


class FliteNarrator(Narrator):
    """Real (if robotic) speech synthesis via `ffmpeg -f lavfi -i flite=textfile=...`."""

    name = "flite"

    def __init__(self, voice: str = "kal", sample_rate: int = 22050) -> None:
        self._voice = voice
        self._sample_rate = sample_rate

    async def is_available(self) -> bool:
        try:
            output = await run_subprocess(
                [settings.ffmpeg_binary, "-hide_banner", "-h", "filter=flite"],
                timeout=10.0,
            )
            return "flite" in output.lower() and "unknown filter" not in output.lower()
        except SubprocessError as e:
            logging.exception(f"ffmpeg flite check failed: {e}")
            return False

    async def synthesize(self, text: str, out_path: Path) -> float:
        # Write narration text to a sibling file and reference it by a bare
        # filename (running ffmpeg with cwd=out_path.parent). This sidesteps
        # ffmpeg lavfi filtergraph escaping entirely (':' and other special
        # characters in the text, or in a Windows drive-letter path, would
        # otherwise break the `flite=text='...'` argument parser).
        text_file = out_path.with_suffix(".narration.txt")
        text_file.write_text(text)

        cmd = [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"flite=textfile={text_file.name}:voice={self._voice}",
            "-ar",
            str(self._sample_rate),
            out_path.name,
        ]
        try:
            await run_subprocess(cmd, cwd=out_path.parent)
        except SubprocessError as exc:
            logging.exception(f"flite synthesis failed: {exc}")
            raise NarrationUnavailableError(f"flite synthesis failed: {exc}") from exc

        with wave.open(str(out_path), "rb") as wav:
            duration = wav.getnframes() / float(wav.getframerate())
        return duration


class PiperNarrator(Narrator):
    """
    Natural-sounding, fully offline neural TTS via the Piper engine
    (https://github.com/rhasspy/piper). Requires the `piper-tts` package
    (`uv sync --extra ai`) and a downloaded voice model:

        piper-tts --model en_US-lessac-medium --data-dir . --download-dir .

    then point `VIDEO_PIPER_MODEL_PATH` at the resulting `.onnx` file. If
    the package or model isn't available, `is_available()` returns False
    and the pipeline falls back to the next backend.
    """

    name = "piper"

    def __init__(self, model_path: str, sample_rate: int = 22050) -> None:
        self._model_path = model_path
        self._sample_rate = sample_rate

    async def is_available(self) -> bool:
        if not self._model_path or not Path(self._model_path).exists():
            return False
        try:
            import piper  # noqa: F401
        except ImportError:
            logging.warning("Piper TTS backend unavailable: 'piper-tts' package not installed")
            return False
        return True

    async def synthesize(self, text: str, out_path: Path) -> float:
        # Shell out to the `piper` CLI (installed alongside the `piper-tts`
        # package) rather than binding to its Python API directly: the CLI
        # surface is small and stable, and keeps this in the same
        # subprocess-based style as the other backends.
        cmd = [
            "piper",
            "--model",
            self._model_path,
            "--output_file",
            out_path.name,
        ]
        process_input = text.encode("utf-8")
        try:
            await run_subprocess_with_stdin(cmd, cwd=out_path.parent, input_bytes=process_input)
            return await probe_duration(out_path)
        except SubprocessError as exc:
            logging.exception(f"Piper synthesis failed: {exc}")
            raise NarrationUnavailableError(f"Piper synthesis failed: {exc}") from exc


class EdgeTTSNarrator(Narrator):
    """
    Natural-sounding cloud neural TTS via Microsoft Edge's free online
    voices, using the `edge-tts` package (`uv sync --extra ai`). This is
    the most realistic-sounding option available here, at the cost of
    needing network access at synthesis time (and depending on an
    unofficial, free API that Microsoft could change or restrict).
    """

    name = "edge-tts"

    def __init__(self, voice: str = "en-US-AndrewNeural") -> None:
        self._voice = voice

    async def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            logging.warning("Edge TTS backend unavailable: 'edge-tts' package not installed")
            return False
        return True

    async def synthesize(self, text: str, out_path: Path) -> float:
        import edge_tts

        mp3_path = out_path.with_suffix(".mp3")
        try:
            communicate = edge_tts.Communicate(text, voice=self._voice)
            await communicate.save(str(mp3_path))
        except Exception as exc:
            logging.exception(f"edge-tts synthesis failed: {exc}")
            # edge-tts's underlying network/websocket errors aren't a type
            # we can enumerate exhaustively (connection refused, DNS
            # failure, timeout, protocol errors, service changes, ...) —
            # normalize anything here to NarrationUnavailableError so a
            # wrapping FallbackNarrator can catch it and cascade to the
            # next backend, exactly the way network failures in
            # AnthropicScriptProvider/AnthropicTopicClassifier already do.
            raise NarrationUnavailableError(f"edge-tts synthesis failed: {exc}") from exc

        # Transcode to WAV so every narrator backend produces a uniform,
        # `wave`-module-readable output regardless of its native codec.
        try:
            await run_subprocess(
                [
                    settings.ffmpeg_binary,
                    "-hide_banner",
                    "-y",
                    "-i",
                    mp3_path.name,
                    "-ar",
                    "22050",
                    out_path.name,
                ],
                cwd=out_path.parent,
            )
            duration = await probe_duration(out_path)
        except SubprocessError as exc:
            logging.exception(f"failed to transcode edge-tts output: {exc}")
            raise NarrationUnavailableError(f"failed to transcode edge-tts output: {exc}") from exc
        finally:
            mp3_path.unlink(missing_ok=True)

        return duration


async def run_subprocess_with_stdin(cmd: list[str], cwd: Path, input_bytes: bytes) -> None:
    """Small variant of run_subprocess that feeds data over stdin (Piper reads text this way)."""
    import asyncio

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate(input=input_bytes)
    if process.returncode != 0:
        output = stdout.decode(errors="replace") if stdout else ""
        logging.error(f"subprocess failed: {' '.join(cmd)}\n{output[-1000:]} error: {process.returncode}")
        raise SubprocessError(
            f"command exited with code {process.returncode}: {' '.join(cmd)}\n{output[-1000:]}"
        )


class FallbackNarrator(Narrator):
    """
    Wraps a primary `Narrator` with a fallback: if the primary raises
    `NarrationUnavailableError` — at *any* synthesis call, not just once
    at startup — transparently use the fallback for that slide instead of
    failing the whole job. Chains of these (built by `select_narrator`)
    are what makes a mid-session network outage for `EdgeTTSNarrator`
    degrade gracefully to Piper/flite/tone and recover automatically once
    the network returns, rather than sticking with one bad startup-time
    choice for the rest of the process's lifetime.
    """

    def __init__(self, primary: Narrator, fallback: Narrator) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}(fallback={fallback.name})"

    async def is_available(self) -> bool:
        return True  # the fallback is always available, so this wrapper always is

    async def synthesize(self, text: str, out_path: Path) -> float:
        try:
            return await self._primary.synthesize(text, out_path)
        except NarrationUnavailableError as e:
            logging.exception(f"primary narrator {self._primary.name} failed, falling back to {self._fallback.name}: {e}")
            return await self._fallback.synthesize(text, out_path)


# Priority order when the caller wants the most authentic voice available
# ("ai" generation provider): try cloud neural TTS first, then offline
# neural TTS, then robotic-but-real offline speech, then the tone fallback.
_REALISTIC_PRIORITY: list[str] = ["edge-tts", "piper", "flite", "tone"]
# The "simulated" generation provider deliberately never reaches for
# network-dependent or extra-heavy backends — fast, offline, reproducible.
_SIMULATED_PRIORITY: list[str] = ["flite", "tone"]

_cache: dict[str, Narrator] = {}


def _build_candidate(backend_name: str) -> Narrator:
    if backend_name == "edge-tts":
        return EdgeTTSNarrator(voice=settings.edge_tts_voice)
    if backend_name == "piper":
        return PiperNarrator(model_path=settings.piper_model_path, sample_rate=settings.tts_sample_rate)
    if backend_name == "flite":
        return FliteNarrator(voice=settings.flite_voice, sample_rate=settings.tts_sample_rate)
    if backend_name == "tone":
        return ToneNarrator(sample_rate=settings.tts_sample_rate)
    raise ValueError(f"unknown narrator backend: {backend_name}")


async def select_narrator(prefer_realistic: bool = False) -> Narrator:
    """
    Build a `FallbackNarrator` chain covering every backend that passes
    its `is_available()` check, in priority order, always ending in
    `ToneNarrator` (which needs no check — it has no external
    dependencies and can't fail). The chain is cached per priority mode
    so repeat calls (e.g. once per worker) don't re-probe every time, but
    — critically — the *fallback behavior itself* is re-evaluated on
    every `synthesize()` call, not just at chain-construction time. This
    matters specifically for `EdgeTTSNarrator`: its `is_available()` can
    only confirm the package is installed, not that the network is
    actually reachable, so it may end up in the chain even when it will
    fail at the next real call — the chain's job is to catch that failure
    when (not if) it happens and cascade to the next backend seamlessly.

    Used by the "simulated" provider (which only ever reaches flite/tone
    — see `_SIMULATED_PRIORITY`). For the "ai" provider's stricter
    behavior, see `select_ai_narrator()` below.
    """
    cache_key = "realistic" if prefer_realistic else "simulated"
    if cache_key in _cache:
        return _cache[cache_key]

    priority = _REALISTIC_PRIORITY if prefer_realistic else _SIMULATED_PRIORITY

    chain: Narrator | None = None
    for backend_name in reversed(priority):
        if backend_name == "tone":
            chain = _build_candidate("tone")  # unconditional final link
            continue
        candidate = _build_candidate(backend_name)
        if not await candidate.is_available():
            continue
        chain = FallbackNarrator(primary=candidate, fallback=chain) if chain is not None else candidate

    if chain is None:  # pragma: no cover - defensive; "tone" always builds a chain above
        chain = ToneNarrator(sample_rate=settings.tts_sample_rate)

    _cache[cache_key] = chain
    return chain


class _NoAIVoiceAvailableNarrator(Narrator):
    """
    Used by `select_ai_narrator()` when neither edge-tts nor Piper is even
    nominally available at startup (no package installed, no model
    configured). Every call fails with a clear, actionable error rather
    than the pipeline silently reaching for flite/tone — the same
    backends the "simulated" provider uses — which would produce a lower
    quality video than "ai" mode implies without the caller ever being
    told that happened.
    """
    name = "no-ai-voice-available"
    async def synthesize(self, text: str, out_path: Path) -> float:
        logging.error("AI voice generation is unavailable: neither edge-tts (needs network "
                      "and the 'edge-tts' package) nor Piper (needs VIDEO_PIPER_MODEL_PATH "
                      "pointing at a downloaded voice model) is currently usable. Install "
                      "and configure at least one (`uv sync --extra ai`), or submit this "
                      "request with GENERATION_PROVIDER=simulated instead.")
        raise NarrationUnavailableError(
            "AI voice generation is unavailable: neither edge-tts (needs network "
            "and the 'edge-tts' package) nor Piper (needs VIDEO_PIPER_MODEL_PATH "
            "pointing at a downloaded voice model) is currently usable. Install "
            "and configure at least one (`uv sync --extra ai`), or submit this "
            "request with GENERATION_PROVIDER=simulated instead."
        )


# "ai" mode's narrator never falls all the way to flite/tone — those are
# what "simulated" mode uses, and silently landing on them would give the
# caller simulated-quality audio while believing they requested "ai".
_AI_STRICT_PRIORITY: list[str] = ["edge-tts", "piper"]


async def select_ai_narrator() -> Narrator:
    """
    Build the "ai" mode narrator: cascades between edge-tts and Piper —
    both genuinely natural-sounding, "real" voice options — on a
    per-call synthesis failure (e.g. network drops mid-session), exactly
    like `select_narrator()`'s general chain. Deliberately does **not**
    fall back further to flite/tone: if neither can produce audio, the
    job fails with a clear `NarrationUnavailableError` instead of
    silently downgrading to simulated-quality voice.
    """
    cache_key = "ai_strict"
    if cache_key in _cache:
        return _cache[cache_key]

    chain: Narrator | None = None
    for backend_name in reversed(_AI_STRICT_PRIORITY):
        candidate = _build_candidate(backend_name)
        if not await candidate.is_available():
            logging.warning(f"{backend_name} backend unavailable")
            continue
        chain = FallbackNarrator(primary=candidate, fallback=chain) if chain is not None else candidate

    if chain is None:
        chain = _NoAIVoiceAvailableNarrator()

    _cache[cache_key] = chain
    return chain
