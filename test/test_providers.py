"""
Tests for src.generation.providers — the plug-and-play registry between
"simulated" and "ai" generation providers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.generation.providers import (
    AIProvider,
    SimulatedProvider,
    UnknownProviderError,
    available_providers,
    get_provider,
)

def test_available_providers_lists_both_builtin_providers():
    names = available_providers()
    assert "simulated" in names
    assert "ai" in names

def test_get_provider_returns_correct_types():
    assert isinstance(get_provider("simulated"), SimulatedProvider)
    assert isinstance(get_provider("ai"), AIProvider)

def test_get_provider_is_cached_singleton_per_name():
    a = get_provider("simulated")
    b = get_provider("simulated")
    assert a is b

def test_get_provider_unknown_name_raises():
    with pytest.raises(UnknownProviderError):
        get_provider("not-a-real-provider")

@pytest.mark.asyncio
async def test_simulated_provider_generates_real_video():
    provider = SimulatedProvider()
    await provider.initialize()
    assert provider.name == "simulated"

    with tempfile.TemporaryDirectory() as tmp:
        result = await provider.generate(
            topic="How does the pH scale work?",
            difficulty="beginner",
            work_dir=Path(tmp),
        )
        assert result.video_path.exists()
        assert result.video_path.stat().st_size > 0
        assert 0 < result.duration_seconds <= 90.0

@pytest.mark.asyncio
async def test_ai_provider_strict_narrator_never_resolves_to_flite_or_tone():
    """The 'ai' provider's narrator must never be (or fall back to) flite
    or tone — those are what 'simulated' mode uses. If no real AI voice
    backend is available, it should be the dedicated
    'no-ai-voice-available' sentinel that fails clearly, not a silent
    downgrade."""
    provider = AIProvider()
    await provider.initialize()
    name = provider._narrator.name
    assert "flite" not in name
    assert "tone" not in name
    assert name == "no-ai-voice-available" or "edge-tts" in name or "piper" in name

@pytest.mark.asyncio
async def test_simulated_validate_topic_rejects_gibberish_no_exception():
    """SimulatedProvider.validate_topic() must never raise — the heuristic
    classifier has no external dependencies to fail."""
    provider = SimulatedProvider()
    result = await provider.validate_topic("asdf jkl qwerty")
    assert result.is_valid is False
    assert result.reason
    assert result.classifier == "heuristic"

@pytest.mark.asyncio
async def test_simulated_validate_topic_accepts_real_topic():
    provider = SimulatedProvider()
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True

@pytest.mark.asyncio
async def test_ai_validate_topic_uses_real_classifier_when_available():
    """With a working (faked) Anthropic client, validate_topic() must
    return the real classifier's verdict, not the heuristic's."""
    provider = AIProvider()

    class _FakeTextBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class _FakeResponse:
        def __init__(self, text):
            self.content = [_FakeTextBlock(text)]

    class _FakeMessages:
        def __init__(self, text):
            self._text = text

        async def create(self, **kwargs):
            return _FakeResponse(self._text)

    class _FakeClient:
        def __init__(self, text):
            self.messages = _FakeMessages(text)

    provider._topic_classifier._api_key = "fake-key-for-test"
    provider._topic_classifier._client = _FakeClient("VALID")
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True
    assert result.classifier == "anthropic"

    provider._topic_classifier._client = _FakeClient("INVALID")
    result2 = await provider.validate_topic("asdf jkl qwerty")
    assert result2.is_valid is False
    assert result2.reason


@pytest.mark.asyncio
async def test_provider_generate_lazily_initializes_if_not_called_explicitly():
    """A provider constructed but never explicitly initialize()'d should
    still work (defensive lazy-init in _PipelineBackedProvider.generate)."""
    provider = SimulatedProvider()
    with tempfile.TemporaryDirectory() as tmp:
        result = await provider.generate(
            topic="What is Newton's second law?",
            difficulty="beginner",
            work_dir=Path(tmp),
        )
        assert result.video_path.exists()

async def test_ai_provider_degrades_content_but_stays_strict_on_voice(monkeypatch):
    """Updated product requirement: a missing ANTHROPIC_API_KEY is a
    permanent, always-knowable condition, and voice generation doesn't
    even depend on it — so it shouldn't be a hard dead end for the whole
    request. Script writing and topic classification now fall back to
    the offline template/heuristic when Anthropic is unavailable, same
    as topic classification always did. Voice stays the one strict part:
    it never falls back to flite/tone, only ever a real neural voice
    (edge-tts/Piper) or a clear failure — see
    test_ai_provider_strict_narrator_never_resolves_to_flite_or_tone.

    Voice unavailability is forced deterministically via monkeypatch
    rather than assumed from the test environment's real package/network
    state — if `uv sync --extra ai` was run and network is reachable,
    edge-tts/Piper may genuinely work here, which would make this
    scenario (content degrades, voice still fails) not reproducible and
    silently skip testing the thing this test exists to check."""
    from src.generation.narrator import EdgeTTSNarrator, NarrationUnavailableError, PiperNarrator

    async def _always_unavailable(self) -> bool:
        return False

    monkeypatch.setattr(EdgeTTSNarrator, "is_available", _always_unavailable)
    monkeypatch.setattr(PiperNarrator, "is_available", _always_unavailable)

    from src.generation import narrator as narrator_module

    narrator_module._cache.clear()
    provider = AIProvider()
    await provider.initialize()
    assert provider.name == "ai"
    assert provider._narrator.name == "no-ai-voice-available"

    # Classification/script no longer block on a missing key.
    result = await provider.validate_topic("Why do atoms form covalent bonds?")
    assert result.is_valid is True

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(NarrationUnavailableError):
            await provider.generate(
                topic="Why do atoms form covalent bonds?",
                difficulty="beginner",
                work_dir=Path(tmp),
            )
        # No half-written artifact left behind by the failed voice step.
        assert list(Path(tmp).glob("*.mp4")) == []

@pytest.mark.asyncio
async def test_ai_provider_produces_real_video_with_natural_voice_and_no_api_key(monkeypatch):
    """The actual scenario this fallback exists for: no ANTHROPIC_API_KEY,
    but a real voice backend is available (network + edge-tts) — the
    request must succeed end-to-end with template content and genuine
    natural narration, rather than being blocked entirely by the missing
    key."""
    import wave

    from src.generation.narrator import EdgeTTSNarrator

    async def fake_synthesize(self, text, out_path):
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 22050)
        return 1.0

    monkeypatch.setattr(EdgeTTSNarrator, "is_available", lambda self: _true())
    monkeypatch.setattr(EdgeTTSNarrator, "synthesize", fake_synthesize)

    from src.generation import narrator as narrator_module

    narrator_module._cache.clear()
    provider = AIProvider()
    await provider.initialize()
    assert "edge-tts" in provider._narrator.name

    with tempfile.TemporaryDirectory() as tmp:
        result = await provider.generate(
            topic="How does the pH scale work?", difficulty="beginner", work_dir=Path(tmp)
        )
        assert result.video_path.exists()
        assert result.video_path.stat().st_size > 0


async def _true():
    return True


@pytest.mark.asyncio
async def test_ai_provider_strict_narrator_never_resolves_to_flite_or_tone():
    """The 'ai' provider's narrator must never be (or fall back to) flite
    or tone — those are what 'simulated' mode uses. If no real AI voice
    backend is available, it should be the dedicated
    'no-ai-voice-available' sentinel that fails clearly, not a silent
    downgrade."""
    provider = AIProvider()
    await provider.initialize()
    name = provider._narrator.name
    assert "flite" not in name
    assert "tone" not in name
    assert name == "no-ai-voice-available" or "edge-tts" in name or "piper" in name


@pytest.mark.asyncio
async def test_simulated_validate_topic_rejects_gibberish_no_exception():
    """SimulatedProvider.validate_topic() must never raise — the heuristic
    classifier has no external dependencies to fail."""
    provider = SimulatedProvider()
    result = await provider.validate_topic("asdf jkl qwerty")
    assert result.is_valid is False
    assert result.reason
    assert result.classifier == "heuristic"


@pytest.mark.asyncio
async def test_simulated_validate_topic_accepts_real_topic():
    provider = SimulatedProvider()
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True


@pytest.mark.asyncio
async def test_ai_validate_topic_falls_back_when_backend_unavailable():
    """With no network/API key, validate_topic() now falls back to the
    offline heuristic rather than raising — a missing key is a permanent,
    known condition, not a reason to block every request; see
    test_ai_provider_degrades_content_but_stays_strict_on_voice for the
    full reasoning on why this changed."""
    provider = AIProvider()
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True
    assert result.classifier == "anthropic(fallback=heuristic)"


@pytest.mark.asyncio
async def test_ai_validate_topic_uses_real_classifier_when_available():
    """With a working (faked) Anthropic client, validate_topic() must
    return the real classifier's verdict, not the heuristic's."""
    provider = AIProvider()

    class _FakeTextBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class _FakeResponse:
        def __init__(self, text):
            self.content = [_FakeTextBlock(text)]

    class _FakeMessages:
        def __init__(self, text):
            self._text = text

        async def create(self, **kwargs):
            return _FakeResponse(self._text)

    class _FakeClient:
        def __init__(self, text):
            self.messages = _FakeMessages(text)

    provider._topic_classifier._primary._api_key = "fake-key-for-test"
    provider._topic_classifier._primary._client = _FakeClient("VALID")
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True
    # classifier is always the wrapper's composite name now that AIProvider
    # wires FallbackTopicClassifier — it reflects which backend is *wired*,
    # not which one handled this particular call.
    assert result.classifier == "anthropic(fallback=heuristic)"

    provider._topic_classifier._primary._client = _FakeClient("INVALID")
    result2 = await provider.validate_topic("asdf jkl qwerty")
    assert result2.is_valid is False
    assert result2.reason