# prep_stt

Whisper (large-v3) Speech-to-Text served over HTTP — the **GPU stage** of the video STT pipeline.
Built for throughput: models are warm-loaded once at startup and stay resident.

[한국어 README](README.ko.md)

## Overview

Given a source video in S3, `prep_stt` extracts the audio, runs the full GPU STT
pipeline, and returns time-stamped, speaker-tagged transcript segments.

The service exposes two stages:

- **`pre_svc`** — S3 → local NVMe download, then `ffmpeg` demux (audio extraction + 6s video chunks). CPU/IO bound.
- **`stt_svc`** — denoise → VAD → LID → ASR → speaker diarization → `result.json`. GPU bound.

## Pipeline

```
video (S3)
   │  pre_svc
[1] download    — S3 → NVMe  (output/{vid}/)
[2] demux       — ffmpeg: extract audio (mono, 16 kHz) + 6s video chunks
   │  stt_svc
[3] denoise     — DeepFilterNet v3            (GPU 0)
[4] VAD         — Silero VAD     (raw audio)
[5] LID         — Whisper detect_language (raw audio, per VAD segment)
[6] ASR         — faster-whisper large-v3 (denoised audio, batched)  (GPU 1)
[7] diarization — pyannote (speaker labels)  (GPU 1)
   └→ result.json  (segments: start/end, text, lang, speaker)
```

**raw / denoised split**: VAD + LID run on the *raw* audio (denoise distorts the LID
signal), while ASR runs on the *denoised* audio (fewer hallucinations). Language-mixed
audio (`ko`/`en` interleaved) is routed per-language, batched, then merged by timestamp.

## Layout

```
config.py            settings — paths / port / models / thresholds (reads .env)
main.py              FastAPI entrypoint — app + routers + GPU warmup
lib/
  http/              routers + request/response DTOs   (pre.py, stt.py)
  service/           business logic, transport-agnostic (pre_service.py, stt_service.py)
  audio/             GPU model components (denoise, vad, def_lang=LID, whisper, speaker)
  io/                S3 download / ffmpeg (CPU)         (s3.py, ffmpeg.py)
  util.py
```

## Requirements

- Python **3.12–3.13**
- [uv](https://github.com/astral-sh/uv)
- **Rust toolchain (`cargo`)** — `deepfilterlib` ships no prebuilt wheel and is compiled from source at install time. Install via `dnf install -y cargo` or [rustup](https://rustup.rs).
- NVIDIA GPU with CUDA — per-model GPU assignment is configurable in `config.py`.
- An NVIDIA-enabled `ffmpeg` build at `/usr/local/ffmpeg-gpu/ffmpeg` (for `FFMPEG_MODE=gpu`; switch to `cpu` to use system ffmpeg).
- Model weights present locally under `$MODELS_ROOT` (see below) — loaded offline, no HF download at runtime.

## Setup

```bash
uv venv .venv --python 3.13
.venv/bin/uv sync
```

### Models

Weights are loaded from fixed local paths under `MODELS_ROOT` (default `/stg/models`):

| Component | Model | Path |
|---|---|---|
| ASR + LID | faster-whisper large-v3 | `$MODELS_ROOT/whisper-large-v3` |
| Denoise | DeepFilterNet v3 | `$MODELS_ROOT/deepfilternet3` |
| VAD | Silero VAD (torch.hub local) | `$MODELS_ROOT/silero-vad` |
| Diarization | pyannote (community-1) | `$MODELS_ROOT/pyannote-diarization` |

### Configuration (`.env`)

```bash
HOST=0.0.0.0
PORT=8080
MODELS_ROOT=/stg/models
S3URL=https://<bucket>.s3.<region>.amazonaws.com   # source video host: GET {S3URL}/{file}
```

`.env` overrides OS environment (`load_dotenv(override=True)`). See `.env.example`.

## Run

### Development

```bash
.venv/bin/python main.py        # serves on $HOST:$PORT, warms up GPU models first (~3s)
```

### Production (systemd)

```bash
sudo cp prep_stt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prep_stt
sudo systemctl status prep_stt
```

A single process is used intentionally — **no** `uvicorn --workers` (each worker would
reload the models and break the GPU queue).

## API

Base URL: `http://$HOST:$PORT`

### `GET /` — health

```json
{ "message": "hello world", "service": "prep_stt" }
```

### `POST /pre_svc/` — download + demux

```bash
curl -X POST http://localhost:8080/pre_svc/ \
  -H 'Content-Type: application/json' \
  -d '{"vid": "6", "file": "6.mp4"}'
```

| Field | Description |
|---|---|
| `vid` | video id — also the job dir name (`output/{vid}/`) |
| `file` | S3 object key — fetched from `{S3URL}/{file}` (extension may not be `.mp4`) |

```json
{
  "status": "ok",
  "job_id": "6",
  "video_path": "output/6/source.mp4",
  "audio_path": "output/6/audio.wav",
  "num_chunks": 42
}
```

### `POST /stt_svc/` — run STT (synchronous)

```bash
curl -X POST http://localhost:8080/stt_svc/ \
  -H 'Content-Type: application/json' \
  -d '{"vid": "6", "file_path": "output/6/audio.wav"}'
```

| Field | Description |
|---|---|
| `vid` | job id, shared with `pre_svc` (`output/{vid}/`) |
| `file_path` | input audio (NVMe path, e.g. `pre_svc`'s `audio.wav`) |

```json
{
  "job_id": "6",
  "status": "done",
  "segments": [
    { "start": "00:00:01.2", "end": "00:00:03.8", "text": "...", "lang": "ko", "speaker": "S001" }
  ],
  "error": ""
}
```

`status` is `"done"` or `"error"` (message in `error`). The call **blocks** until the
pipeline finishes — RTF ≈ 0.042, so a 1-hour input takes ~150 s, 6 hours ~15 min.
The full transcript is also written to `output/{vid}/result.json`.

> The HTTP layer is synchronous today; an async `202 + polling` variant may be added
> for very long inputs.

## Output

NVMe scratch, one directory per job (the durable copy belongs in S3):

```
output/{vid}/
├── source.mp4      # downloaded original
├── audio.wav       # extracted mono 16 kHz
├── denoise.wav     # (only if SAVE_STEPS["denoise"] = True)
└── result.json     # final segments
```
