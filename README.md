# STEM Explainer Video Service (FastAPI)

A FastAPI service which generates a video to explain a STEM topic submitted by users.

- "How does the pH scale work?"
- "Why do atoms form covalent bonds?"
- "What is the difference between ionic and covalent bonding?"

— and gets back a short explainer video: real generated slide visuals
**and** narrated audio, like a short educational video, produced
**asynchronously** in the background and capped at **90 seconds**. The
generation backend itself is **pluggable** between a fully offline
simulated pipeline and a real-AI-backed one, without touching any other
layer of the app.

This project is generated using Claude desktop in multiple rounds of prompts.

## Prompts

```
Create a Python FastAPI service with an API endpoint where a client can request a STEM concept explanation video. STEM stands for Science, Technology, Engineering and Mathematics. The service must be designed and implemented using asynchronous video-generation flow. Here are the requirements for the application:

1. Please use uv package manager.
2. Clear service design which enables plug-and-play of both simulated generation and real AI/video-generation providers.
3. src/ folder should contain the application source code.
4. test/ folder should contain the tests.
5. output/ folder should contain the outputs of the application. For example, the generated videos and the job statuses.
6. a way for learners to submit their query of the topic they are interested in. Here are some examples of the questions that the application must be able to answer:
   i. How does the pH scale work?
  ii. Why do atoms form covalent bonds?
 iii. What is the difference between ionic and covalent bonding?
7. a way to list requested videos or jobs
8. a visible status for each requested video or job
9. a way to retrieve or open a completed video explanation artifact
10. visual content and audio for the explanation, similar to how a normal short educational video would feel
11. a clear backend boundary for job state, generation logic, persistence, and artifacts
12. Limit video length to 90 seconds.
13. Use 4K video resolution.
14. Use 60 fps.
15. Use Nvidia GPU for hardware acceleration.
16. The voice of the video should be as authentic and real as possible as compared to monotonic synthetic voice.
17. Scale the number of slides generated according to difficulty level.

Failure cases which should NOT fallback to simulated generation but to return a proper error status and message to the user immediately without creating any job:
1. Invalid query string even if the length is right. For example:
   i.  ~!@#$
  ii. 12345
 iii. "     "
2. Questions which do not make any semantic sense or not being relevant to STEM topic.
3. No network or improper API key in "ai" generation mode.
```

## Architecture

```
src/
  main.py                  API layer (FastAPI routes)
  worker.py                  Orchestration seam — talks to state, artifacts, and
                              whichever generation provider is configured

  state/                    ── JOB STATE ──────────────────────────────────
    job_state.py               Lifecycle rules: valid transitions, progress,
                                cancellation, startup recovery.

  persistence/                ── PERSISTENCE ────────────────────────────────
    repository.py               Durable storage of job metadata (JSON files
                                 under output/jobs/ today; swappable backend).

  artifacts/                   ── ARTIFACTS ──────────────────────────────────
    store.py                     Storage/retrieval of the generated video
                                  FILES (output/videos/) — a distinct concern
                                  from job metadata.

  generation/                   ── GENERATION LOGIC (pluggable) ──────────────
    providers.py                  Provider registry/factory: "simulated" vs "ai"
    topic_classifier.py            TopicClassifier interface: HeuristicTopicClassifier
                                    (offline) vs AnthropicTopicClassifier (real LLM)
    script_providers.py             ScriptProvider interface: SimulatedScriptProvider
                                     (template) vs AnthropicScriptProvider (real LLM)
    narrator.py                      Narrator interface: ToneNarrator, FliteNarrator,
                                      PiperNarrator, EdgeTTSNarrator
    slide_renderer.py                 Slide -> PNG image (Pillow)
    video_assembler.py                 Slide images + audio -> final MP4 (ffmpeg)
    pipeline.py                       Orchestrates the above; enforces the 90s cap
```

Each layer only knows its own job — state enforces lifecycle rules,
persistence just durably stores job records, artifacts just stores/serves
video files, and generation turns a topic into a video file with zero
knowledge of jobs, HTTP, or storage. `worker.py` is the only place all of
them meet.

### Plug-and-play generation providers

This is the direct answer to "clear service design which enables
plug-and-play of both simulated generation and real AI/video-generation
providers": `src/generation/providers.py` defines a `GenerationProvider`
interface with two registered implementations, selected via the
`GENERATION_PROVIDER` environment variable:

|                             | `simulated` (default)                                                                                | `ai`                                                                                                                                                                                                                                                                           |
| --------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Script writer**           | `SimulatedScriptProvider` — deterministic template, no API calls                                     | `AnthropicScriptProvider` — real Claude API call, genuinely answers the learner's question. Falls back to the template if `ANTHROPIC_API_KEY` is unset or the call fails, since a missing key is a permanent, always-knowable condition and shouldn't block the whole request. |
| **Voice**                   | `flite` (offline, robotic-but-real), falling back to a tone track                                    | A real AI voice only: cloud neural TTS (`edge-tts`) or offline neural TTS (`Piper`). **No fallback** to `flite`/tone: if neither works, rejected immediately rather than silently substituting simulated-quality voice.                                                        |
| **Topic classification**    | `HeuristicTopicClassifier` — offline pattern-matching, checked synchronously before a job is created | `AnthropicTopicClassifier` — real LLM call, understands meaning, checked synchronously before a job is created. Falls back to the heuristic if unavailable, same reasoning as the script writer.                                                                               |
| **Network/API keys needed** | None                                                                                                 | Nothing strictly required — script/classification degrade gracefully without `ANTHROPIC_API_KEY`; only voice generation needs network (for `edge-tts`) or a configured Piper model to produce a video at all                                                                   |
| **Used by**                 | The test suite; local dev without any setup                                                          | Deployments that want accurate, natural-sounding videos, with network as the one hard dependency                                                                                                                                                                               |

**Content vs. voice are treated differently on purpose.** A missing
`ANTHROPIC_API_KEY` is permanent and known in advance — refusing every
request over it forever, when voice generation doesn't even depend on
that key, would make `ai` mode a dead end for no good reason. So script
writing and topic classification fall back to the offline
template/heuristic when Anthropic is unavailable (no key, network down,
timeout). Voice is different: it's the one thing that's supposed to set
`ai` mode apart audibly, so it stays strict — `select_ai_narrator()` only
ever resolves to a genuine neural voice (`edge-tts`/Piper) or fails
clearly, never silently dropping to `flite`/tone. Net effect: with a
working `ANTHROPIC_API_KEY`, you get real AI-authored content **and**
natural voice; with just network and no key, you still get natural voice
over template content, which beats a hard refusal; with neither, the
request fails clearly rather than silently producing `simulated`-quality
output. See "Failure handling" below for the exact behavior and HTTP
status per scenario.

Both providers share the same slide renderer and video assembler — swap
`GENERATION_PROVIDER=simulated` for `GENERATION_PROVIDER=ai` and nothing
else in the app changes. Adding a third provider (e.g. a real
generative-video API instead of the local Pillow/ffmpeg renderer) means
implementing the same `GenerationProvider` interface and registering it —
`worker.py` and the API layer need no changes.

Check what's active and registered via `GET /api/v1/providers`.

### How a video actually gets made

1. The configured `ScriptProvider` turns the learner's topic/question into
   a script whose **slide count scales with `difficulty`**: 5 slides for
   `beginner` (title, what it is, why it matters, key takeaway, summary),
   7 for `intermediate` (adds a two-part "how it works" mechanism
   walkthrough), 10 for `advanced` (further adds a common misconception,
   a real-world application, and how the topic connects to related
   ideas). This isn't arbitrary — see "Why slide count scales with
   difficulty" below for the reasoning and the actual numbers behind it.
2. For each slide: `slide_renderer.py` draws a 720p PNG with Pillow, and
   the configured `Narrator` synthesizes speech audio for the narration line.
3. `video_assembler.py` encodes each (image, audio) pair into a video
   segment via ffmpeg, then concatenates all segments into one MP4.
4. If the assembled video would run over 90 seconds, `pipeline.py` trims
   it down to exactly 90s as a hard safety net — narration word counts
   per difficulty tier are deliberately kept under this in the common
   case (see below), so this is a safety net, not the normal path.
5. `worker.py` saves the result via the artifact store (`output/videos/`)
   and marks the job `completed` with size/duration.

### Why slide count scales with difficulty

Originally every request produced a fixed 5-slide script regardless of
`difficulty`, using only ~53s of the 90s budget (~42% unused) — and 5
slides isn't enough room for real depth on a complex topic: barely space
for title/what/why/takeaway/summary, no room for a worked mechanism, a
common misconception, or how the topic connects to related ideas.

Slide count and narration length now scale with `difficulty`
(`_DIFFICULTY_SLIDE_COUNT` / `_DIFFICULTY_WORD_BUDGET` in
`script_providers.py`):

| Difficulty     | Slides | Measured duration (real flite narration)           |
| -------------- | ------ | -------------------------------------------------- |
| `beginner`     | 5      | ~40s                                               |
| `intermediate` | 7      | ~50s                                               |
| `advanced`     | 10     | ~77s (with a real safety margin below the 90s cap) |

For `SimulatedScriptProvider`, the extra slides at higher tiers are
genuinely new content — a two-part mechanism walkthrough, a common
misconception, a real-world application, how the topic connects to
related ideas — not the same five slides padded with more words (every
`beginner` slide heading still appears in `advanced`; advanced just adds
more on top, verified in `test_slide_count_scales_with_difficulty` /
`test_higher_difficulty_slides_are_not_just_beginner_slides_repeated`).
For `AnthropicScriptProvider`, the target slide count and word budget are
passed explicitly to the model in the prompt, so "advanced" reliably gets
more depth rather than leaving it to the model's own judgment.

Even at `advanced`'s 10 slides, real measured narration (~77s) leaves a
genuine margin below the 90s cap — narration length is estimated at a
145 words/minute pace, and real TTS timing can vary slightly, so this
margin matters more than it might look on paper.

### Long topics don't fit in a slide title — two-layer fix

`topic` can be up to 1024 characters (see "Failure handling" below), but
a slide is only 1280px wide — a long query rendered verbatim as a
heading runs straight off the edge of the frame, cut off mid-word and
unreadable. This is fixed at two independent layers:

1. **`_short_title()`** (`script_providers.py`): `SimulatedScriptProvider`
   derives a concise version of the topic — truncated at a word boundary
   with an ellipsis, not mid-word — for use in the title slide's heading
   and the summary slide's body text specifically. **Narration always
   uses the full topic**, unshortened — spoken audio has no width
   constraint, so there's no reason to lose any of the learner's actual
   question there.
2. **Text fitting in `slide_renderer.py`** (`_wrap_to_lines` /
   `_fit_lines` / `_fit_single_line`): an independent safety net that
   wraps _any_ heading it's given to up to 2 lines (growing the header
   band's height to fit), truncating with an ellipsis only if it still
   doesn't fit — and single-line-truncates body text the same way. This
   doesn't know or care whether the text came from `_short_title()`, a
   fixed template string, or an LLM-generated heading (which is only
   _asked_ to stay under 6 words, not guaranteed to) — it fits whatever
   it's handed. It also correctly hard-breaks a pathological single
   "word" with no spaces at all (verified in
   `test_slide_renderer_single_unbreakable_word_does_not_overflow`),
   which word-boundary wrapping alone can't handle.

Both layers are necessary: layer 1 keeps the _common_ case (a long but
normal question) looking clean with a real ellipsis-truncated summary at
a sensible length; layer 2 guarantees nothing ever visually overflows
regardless of what text reaches the renderer, from any source.

## Voice authenticity

Requirement: "the voice should be as authentic and real as possible
compared to a monotonic synthetic voice." `narrator.py` implements two
different selection strategies depending on the generation provider:

**`ai` mode** (`select_ai_narrator()`) only ever uses a genuinely natural
voice — no silent downgrade to anything robotic:

1. **`edge-tts`** (cloud neural TTS, Microsoft Edge's free online voices)
   — the most natural-sounding option here. Needs network at synthesis
   time and the `edge-tts` package (`uv sync --extra ai`).
2. **Piper** (offline neural TTS) — genuinely natural-sounding and fully
   local after a one-time setup. Needs the `piper-tts` package
   (`uv sync --extra ai`) _and_ a downloaded voice model:
   ```bash
   pip install piper-tts   # or: uv run pip install piper-tts
   python -m piper.download_voices en_US-lessac-medium
   "VIDEO_PIPER_MODEL_PATH": "/path/to/en_US-lessac-medium.onnx"
   ```

If **neither** works, the job fails with a clear
`NarrationUnavailableError` instead of silently falling back to flite or
a tone track — a caller who explicitly requested `ai` mode should never
unknowingly receive `simulated`-quality audio. See "Failure handling"
below.

**`simulated` mode** (`select_narrator(prefer_realistic=False)`) never
reaches for edge-tts/Piper at all — it's meant to be fast, free, and
fully offline:

3. **flite** (ffmpeg's built-in `flite` speech filter) — real speech, but
   old-school diphone synthesis, noticeably more robotic. No setup beyond
   ffmpeg itself.
4. **Tone fallback** — if literally nothing else is available (e.g. an
   ffmpeg build without flite support), a dependency-free paced tone
   track keeps the pipeline from failing outright.

Check the startup log line `Generation provider '...' using script
provider '...', topic classifier '...', and narrator '...'` to see which
backend actually got selected for whichever provider is configured.

**Resilience note**: `edge-tts`'s `is_available()` check can only confirm
the package is installed — it can't confirm the network is actually
reachable, since that can change at any moment during a long-running
process. Rather than trust a one-time startup check, both selectors
wrap their chain in `FallbackNarrator`, which catches a real synthesis
failure (network down, connection refused, DNS failure, ...) on **every**
call. In `ai` mode this cascades from edge-tts to Piper (both real AI
voices) and automatically goes back to trying `edge-tts` again on the
next call, with no restart needed — it only reaches the clear
`NarrationUnavailableError` failure described above if Piper isn't
configured either. See "Failure handling" below for the full trace of
what happens with no network in `ai` mode.

## API

| Method | Path                               | Description                                                                                                          |
| ------ | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| POST   | `/api/v1/videos`                   | Submit a topic/question → `202` + `job_id`, or an immediate `422`/`503` with no job created — see "Failure handling" |
| GET    | `/api/v1/videos`                   | **List** all requested videos/jobs, newest first, each with a **visible status**                                     |
| GET    | `/api/v1/videos/{job_id}`          | Status/progress of one job                                                                                           |
| GET    | `/api/v1/videos/{job_id}/download` | **Open/retrieve** the finished video (streams the real `.mp4`)                                                       |
| DELETE | `/api/v1/videos/{job_id}`          | Cancel a pending/in-progress job                                                                                     |
| GET    | `/api/v1/providers`                | List registered generation providers and which is active                                                             |
| GET    | `/health`                          | Liveness probe                                                                                                       |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/videos \
  -H "Content-Type: application/json" \
  -d '{"topic": "How does the pH scale work?", "difficulty": "beginner"}'
# -> {"job_id": "...", "status": "pending", "provider": "simulated", "poll_url": "/api/v1/videos/..."}

curl http://localhost:8000/api/v1/videos/<job_id>
# -> {"status": "processing", "progress": 46, "stage": "rendering_slide_3_of_5", ...}
# ... poll again later ...
# -> {"status": "completed", "progress": 100,
#     "artifact": {"content_type": "video/mp4", "size_bytes": 812344, "duration_seconds": 47.4}}

curl http://localhost:8000/api/v1/videos               # list all jobs + statuses
curl -OJ http://localhost:8000/api/v1/videos/<job_id>/download   # save the mp4
# or just open http://localhost:8000/api/v1/videos/<job_id>/download in a browser tab
```

## Configuration via /etc/fastapi-video-service_config.json

All settings (see the table below) can be set via a local `/etc/fastapi-video-service_config.json` file
instead of exporting environment variables by hand — it's loaded
automatically at startup via [python-dotenv](https://pypi.org/project/python-dotenv/).

All API keys are set in `.env` locally or through secrets when deployed in k8s.

A ready-to-edit config is included (defaults to `GENERATION_PROVIDER=simulated`);
`fastapi-video-service_config.json.example` documents every available option. `.env` is gitignored, so
it's a safe place to put secrets like `ANTHROPIC_API_KEY` locally — real
environment variables (e.g. ones set by a deploy platform) always take
precedence over whatever's in `.env`.

```bash
cp fastapi-video-service_config.json.example /etc/fastapi-video-service_config.json   # if you don't already have one
# edit /etc/fastapi-video-service_config.json to set GENERATION_PROVIDER=ai and your ANTHROPIC_API_KEY, etc.
```

## Running with uv

```bash
uv sync                                  # base install: fully offline "simulated" provider
uv run uvicorn src.main:app --reload
```

To use the real-AI provider, set `GENERATION_PROVIDER=ai` in '/etc/fastapi-video-service_config.json' and
set `ANTHROPIC_API_KEY` in your `.env` file (recommended — see above), or
export them directly:

```bash
uv sync --extra ai                       # installs anthropic, edge-tts, piper-tts
export ANTHROPIC_API_KEY=sk-ant-...      # for real, question-specific scripts
uv run uvicorn src.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.

## Running tests

```bash
uv sync --group dev
uv run pytest
```

Tests run against the **real** generation pipeline (genuine ffmpeg
encoding, genuine speech synthesis) using the `simulated` provider — no
API keys or network required. The `ai` provider has its own dedicated
tests (`test_providers.py`) that verify its graceful-degradation guarantee
(a real video comes out the other end whether or not `ANTHROPIC_API_KEY`
happens to be configured).

## Output

Per the requirement that the app's outputs live under `output/`:

```
output/
  jobs/       one JSON file per job — the durable, visible job status record
  videos/     one .mp4 per completed job — the retrievable video artifact
```

## GPU hardware acceleration (NVIDIA)

Video encoding can use NVIDIA's NVENC hardware encoder instead of CPU-only
libx264, controlled by `VIDEO_HW_ACCEL`:

- **`auto`** (default) — at startup, attempts a real, tiny `h264_nvenc`
  encode to check whether a working NVIDIA GPU is actually available, and
  uses it if so. Falls back to CPU (libx264) automatically and silently
  otherwise — nothing needs configuring if you don't have a GPU.
- **`cpu`** — always use libx264, skip GPU detection entirely.

```bash
# /etc/fastapi-video-service_config.json — usually you don't need to set this at all; "auto" already does
# the right thing whether or not a GPU is present.
"VIDEO_HW_ACCEL": "auto"
```

**Why a real encode, not just a feature check**: an ffmpeg build can have
NVENC support compiled in while the machine has no NVIDIA GPU or driver
installed — in that case `h264_nvenc` shows up in `ffmpeg -encoders` but
fails at encode time (`Cannot load libcuda.so.1`). `hw_accel.py` detects
this correctly by actually attempting a 1-frame encode rather than trusting
the feature list, the same principle used for flite speech-backend
detection.

**GPU requirements**: an NVIDIA GPU, the NVIDIA driver, and an ffmpeg build
with NVENC support (most distro packages and the official static builds
include it). Check `ffmpeg -encoders | grep nvenc` to confirm your ffmpeg
build has it compiled in; `nvidia-smi` to confirm a driver is installed.

Check which encoder actually got selected via the startup log or by
inspecting a generated file: `ffprobe -show_entries stream=codec_name
<file>.mp4` will show `h264` either way (NVENC and libx264 both produce
standard H.264 streams) — the meaningful difference is encoding _speed_,
not the resulting codec.

## Generating 4K video

Set `VIDEO_WIDTH=3840` and `VIDEO_HEIGHT=2160` (in `/etc/fastapi-video-service_config.json` or as environment
variables) to render at 4K instead of the 720p default. The slide layout
(text, diagrams, footer) scales proportionally with resolution — 4K
output is a genuinely crisp, properly laid-out 4K frame, not just a
720p-sized layout stretched onto a bigger canvas.

```bash
# /etc/fastapi-video-service_config.json
"VIDEO_WIDTH": 3840
"VIDEO_HEIGHT": 2160
```

**Performance note**: this repo's ffmpeg build does CPU-only H.264
encoding (no hardware acceleration), so both resolution and frame rate
directly affect encoding time. In local testing with the actual default
settings (720p @ 60fps), a ~48-second video took about 25 seconds to
generate; at 4K @ 24fps, the same script took roughly 2.5 minutes
(~20-30x slower than 720p). 4K combined with 60fps compounds both costs —
expect it to be noticeably slower than either change alone, since a 60fps
encode processes 2.5x as many frames as 24fps at the same resolution and
duration. Each job also ties up a worker for that whole duration, so
consider lowering `VIDEO_NUM_WORKERS` (and/or `VIDEO_FPS`) at 4K to avoid
an accumulating backlog under load. If an NVIDIA GPU is available, set
`VIDEO_HW_ACCEL=auto` (the default) to substantially cut this encoding
time — see "GPU hardware acceleration" below. File sizes stay modest even
at high resolution/frame rate (the slide content is flat, simple graphics
that compress efficiently), but generation latency is the real cost to
budget for.

## System requirements

- **ffmpeg** on `PATH`, used for video/audio encoding and (via its
  built-in `flite` filter) baseline speech synthesis. Verify flite support
  with `ffmpeg -h filter=flite`; Debian/Ubuntu's `apt install ffmpeg`
  includes it.
- **Optional, for the most authentic voice and real AI scripts**:
  `uv sync --extra ai` (installs `anthropic`, `edge-tts`, `piper-tts`) plus
  a Piper voice model download and/or an `ANTHROPIC_API_KEY` — see "Voice
  authenticity" above.

## Configuration (/etc/fastapi-video-service_config.json)

| Variable                                       | Default                     | Meaning                                                                                            |
| ---------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------- |
| `GENERATION_PROVIDER`                          | `simulated`                 | `simulated` or `ai` — which generation provider to use                                             |
| `VIDEO_NUM_WORKERS`                            | `3`                         | Number of concurrent background workers                                                            |
| `VIDEO_JOBS_DIR`                               | `output/jobs`               | Durable job metadata storage directory                                                             |
| `VIDEO_ARTIFACTS_DIR`                          | `output/videos`             | Generated video file storage directory                                                             |
| `VIDEO_MAX_DURATION_SECONDS`                   | `90`                        | Hard cap on generated video length                                                                 |
| `VIDEO_MIN_TOPIC_LENGTH`                       | `5`                         | Minimum accepted length of `topic`                                                                 |
| `VIDEO_WIDTH` / `VIDEO_HEIGHT`                 | `1280` / `720`              | Output video resolution                                                                            |
| `VIDEO_FPS`                                    | `60`                        | Output video frame rate                                                                            |
| `VIDEO_FLITE_VOICE`                            | `kal`                       | flite voice name (`kal`, `awb`, `rms`, `slt`, ...)                                                 |
| `VIDEO_EDGE_TTS_VOICE`                         | `en-US-AndrewNeural`        | edge-tts voice name                                                                                |
| `VIDEO_PIPER_MODEL_PATH`                       | _(unset)_                   | Path to a downloaded Piper `.onnx` voice model                                                     |
| `ANTHROPIC_API_KEY`                            | _(unset)_                   | Required for `AnthropicScriptProvider`                                                             |
| `VIDEO_ANTHROPIC_MODEL`                        | `claude-sonnet-5`           | Anthropic model used for script generation                                                         |
| `VIDEO_ANTHROPIC_CLASSIFIER_MODEL`             | `claude-haiku-4-5-20251001` | Anthropic model used for topic classification (deliberately smaller/cheaper than the script model) |
| `VIDEO_HW_ACCEL`                               | `auto`                      | `auto` (use GPU if detected) or `cpu` (always CPU)                                                 |
| `VIDEO_FFMPEG_BINARY` / `VIDEO_FFPROBE_BINARY` | `ffmpeg` / `ffprobe`        | Paths to the binaries                                                                              |

## Job lifecycle

```
pending → processing → completed
                      ↘ failed
   (any non-terminal) → cancelled
```

- Jobs still `pending`/`processing` when the process restarts are marked
  `failed` at startup rather than left stuck forever.
- Cancelling a running job cancels its underlying task directly;
  cancelling a job still in the queue marks it `cancelled` so a worker
  skips it when dequeued, instead of silently processing it anyway.

## Failure handling

Three categories of failure are rejected **synchronously, with an
immediate error response, and no job ever created** — not discovered
later by polling a job that was doomed from the start. This is enforced
in two layers: pydantic (`src/models.py`) for malformed input, and
`GenerationProvider.validate_topic()` (`src/generation/providers.py`),
called from `POST /api/v1/videos` _before_ `state.create()`, for
everything else.

### 1. Malformed input, even if the length is technically right

Enforced by pydantic in `src/models.py`, before the endpoint body even
runs:

| Input                                                                | Result                                                                                                                                                                                 |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Blank/whitespace-only `topic` (e.g. `"     "`)                       | `422` — rejected even though the _raw_ string meets the length minimum, because validation strips first, then checks                                                                   |
| `topic` with no letters at all (e.g. `"~!@#$"`, `"12345"`, `"...."`) | `422` — a real STEM question always contains at least one letter; this catches pure-symbol/digit garbage without rejecting legitimate short topics like `"pH scale"` or `"DNA repair"` |
| `topic` shorter than `VIDEO_MIN_TOPIC_LENGTH` (default 5)            | `422`                                                                                                                                                                                  |
| `topic` longer than 1024 characters                                  | `422` (pydantic `max_length`)                                                                                                                                                          |
| Invalid `difficulty` (not beginner/intermediate/advanced)            | `422`                                                                                                                                                                                  |
| `duration_seconds` outside 15-90                                     | `422`                                                                                                                                                                                  |

None of these create a job — FastAPI validates the request body against
`VideoRequest` before `request_video()`'s body runs at all.

### 2. Semantically nonsensical or non-STEM topics

`"asdf jkl qwerty"` has letters and a normal length, so it passes every
check above — this is where `src/generation/topic_classifier.py` comes
in. `POST /api/v1/videos` calls `provider.validate_topic(topic)`
**synchronously, before creating a job**:

- **`simulated` provider** → `HeuristicTopicClassifier`, fully offline and
  free, can never fail due to unavailability. Flags QWERTY keyboard-walks
  (`asdf`, `qwerty`, `jkl`, `zxcvbn`), low-character-variety runs
  (`aaaaaa`, `ababab`), and long vowel-less tokens, while exempting
  anything that looks like real technical shorthand (contains a
  digit/symbol, or is a short acronym) so it doesn't false-positive on
  things like `"CRISPR-Cas9"`, `"pH scale"`, `"5G networks"`, or
  `"E=mc2"`. Verified against 20 real STEM topics and 6 gibberish
  variants with zero false positives/negatives. It's a pattern detector,
  not real language understanding, so it won't catch nonsense made of
  real words strung together meaninglessly (e.g. `"purple democracy runs
quickly"`) — a video would still get generated for wording like that.
- **`ai` provider** → `AnthropicTopicClassifier` wrapped in
  `FallbackTopicClassifier`, so a real LLM call judges meaning when
  Anthropic is reachable, transparently falling back to the same offline
  heuristic above when it isn't (see category 3 below for why that
  changed from an earlier, stricter design).

A rejected topic returns `422` with the classifier's reason in `detail`
and creates **no job** — confirmed in `test_main.py`
(`test_gibberish_topic_rejected_immediately_with_no_job_created`), which
checks the job list is identical before and after the rejected
submission.

### 3. No network or an invalid API key in `ai` mode

This used to be a hard `503` at submission time for any Anthropic
failure. It isn't anymore, for topic classification and script writing
specifically — here's the current behavior and the reasoning:

- **Topic classification and script writing** (`AnthropicTopicClassifier`,
  `AnthropicScriptProvider`) are wrapped in `FallbackTopicClassifier` /
  `FallbackScriptProvider`. A missing `ANTHROPIC_API_KEY` is a permanent,
  always-knowable condition — unlike a transient network blip — so rather
  than reject every request over it, these two components transparently
  fall back to the offline template/heuristic. Practical consequence:
  **`POST /api/v1/videos` in `ai` mode essentially never returns a
  synchronous `503` for a missing/invalid key anymore** — `validate_topic()`
  always succeeds via one classifier or the other, so submission proceeds
  to job creation. Verified in `test_providers.py`
  (`test_ai_validate_topic_falls_back_when_backend_unavailable`).
- **Voice** (`select_ai_narrator()`) is the one part that stays strict —
  see "Voice authenticity" above for why. Since voice is only resolved
  during actual job processing (not in the synchronous `validate_topic()`
  pre-check), its failure surfaces differently: the job is created
  (`202`), then fails asynchronously with `status: "failed"` and a clear
  `NarrationUnavailableError` message once generation reaches the
  narration step — not an immediate rejection at submission time.
  Verified in `test_providers.py`
  (`test_ai_provider_degrades_content_but_stays_strict_on_voice`,
  `test_ai_provider_strict_narrator_never_resolves_to_flite_or_tone`).
- **The scenario this all exists for**: no `ANTHROPIC_API_KEY`, but
  network reachable and `edge-tts` installed (`uv sync --extra ai`) — the
  request succeeds end-to-end with template content and genuine natural
  narration, rather than being blocked entirely by a missing key that
  voice generation never needed in the first place. Verified in
  `test_providers.py`
  (`test_ai_provider_produces_real_video_with_natural_voice_and_no_api_key`).

Net result: in `ai` mode, expect a submission-time `422`/`503` only for a
genuinely invalid topic or a completely broken environment (no
Anthropic, no `edge-tts`, no Piper — nothing at all to work with); expect
an async job failure specifically when content generation works
(fallback) but no real voice backend is reachable.

### Generation pipeline failures (`src/worker.py`)

Any exception during script writing, narration, rendering, or encoding —
`GenerationError`, `ProviderUnavailableError`, `NarrationUnavailableError`,
or anything unexpected — is caught in `_process_job()`, logged, and the
job is marked `failed` with the error message visible in
`GET /api/v1/videos/{job_id}`. A crash in one job never takes down its
worker: `worker.py` runs each job as its own `asyncio.Task`, separate
from the worker loop task that dequeued it, so a failure (or
cancellation) only affects that one job.

### Duplicate/repeated job submission

There's no deduplication — submitting the same `topic` twice creates two
independent jobs with different `job_id`s. If that's undesirable for your
use case, dedup would need to be added at the API layer (e.g. hashing
`topic`+`difficulty` and checking `JobStateManager.list_all()` before
creating a new job).

## Continuous Integration:

- Integrated with CircleCI
