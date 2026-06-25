# prep_stt

Whisper(large-v3) 기반 STT 를 HTTP 로 제공하는 서비스 — 영상 STT 파이프라인의 **GPU 단계**.
속도가 최우선: 모델은 시작 시 1회 warmup 으로 로드해 상주시킨다.

[English README](README.md)

## 개요

S3 의 원본 영상을 받아 음성을 추출하고, GPU STT 파이프라인을 끝까지 돌려
타임스탬프 + 화자 태그가 붙은 자막 세그먼트를 반환한다.

두 단계를 제공한다:

- **`pre_svc`** — S3 → 로컬 NVMe 다운로드 후 `ffmpeg` demux (오디오 추출 + 6초 영상청크). CPU/IO bound.
- **`stt_svc`** — denoise → VAD → LID → ASR → 화자 구분 → `result.json`. GPU bound.

## 파이프라인

```
영상 (S3)
   │  pre_svc
[1] download    — S3 → NVMe  (output/{vid}/)
[2] demux       — ffmpeg: 오디오 추출 (mono, 16 kHz) + 6초 영상청크
   │  stt_svc
[3] denoise     — DeepFilterNet v3            (GPU 0)
[4] VAD         — Silero VAD     (raw audio)
[5] LID         — Whisper detect_language (raw audio, VAD segment 단위)
[6] ASR         — faster-whisper large-v3 (denoised audio, 배치)  (GPU 1)
[7] diarization — pyannote (화자 라벨)  (GPU 1)
   └→ result.json  (segments: start/end, text, lang, speaker)
```

**raw / denoised 분리**: VAD + LID 는 *raw* audio 에서 수행하고(denoise 는 LID 신호를
변형시킴), ASR 만 *denoised* audio 에서 수행한다(환각 감소). 언어 혼재 오디오
(`ko`/`en` 교차)는 언어별로 라우팅·배치한 뒤 timestamp 로 병합한다.

## 구조

```
config.py            설정 — 경로 / 포트 / 모델 / 임계값 (.env 로딩)
main.py              FastAPI 진입점 — app + 라우터 + GPU warmup
lib/
  http/              라우터 + 요청/응답 DTO            (pre.py, stt.py)
  service/           비즈니스 로직 (transport 무관)     (pre_service.py, stt_service.py)
  audio/             GPU 모델 컴포넌트 (denoise, vad, def_lang=LID, whisper, speaker)
  io/                S3 다운로드 / ffmpeg (CPU)         (s3.py, ffmpeg.py)
  util.py
```

## 요구사항

- Python **3.12–3.13**
- [uv](https://github.com/astral-sh/uv)
- **Rust 툴체인 (`cargo`)** — `deepfilterlib` 는 미리 빌드된 wheel 이 없어 설치 시 소스에서 컴파일됨. `dnf install -y cargo` 또는 [rustup](https://rustup.rs) 으로 설치.
- CUDA 지원 NVIDIA GPU — 모델별 GPU 배치는 `config.py` 에서 설정.
- `/usr/local/ffmpeg-gpu/ffmpeg` 에 NVIDIA 빌드 ffmpeg (`FFMPEG_MODE=gpu` 용; 시스템 ffmpeg 를 쓰려면 `cpu` 로 전환).
- `$MODELS_ROOT` 아래 모델 가중치 로컬 존재 — 런타임에 HF 다운로드 없이 오프라인 로드.

## 설치

```bash
uv venv .venv --python 3.13
.venv/bin/uv sync
```

### 모델

가중치는 `MODELS_ROOT`(기본 `/stg/models`) 아래 고정 경로에서 로드한다:

| 컴포넌트 | 모델 | 경로 |
|---|---|---|
| ASR + LID | faster-whisper large-v3 | `$MODELS_ROOT/whisper-large-v3` |
| Denoise | DeepFilterNet v3 | `$MODELS_ROOT/deepfilternet3` |
| VAD | Silero VAD (torch.hub local) | `$MODELS_ROOT/silero-vad` |
| 화자 구분 | pyannote (community-1) | `$MODELS_ROOT/pyannote-diarization` |

### 설정 (`.env`)

```bash
HOST=0.0.0.0
PORT=8080
MODELS_ROOT=/stg/models
S3URL=https://<bucket>.s3.<region>.amazonaws.com   # 원본 영상 호스트: GET {S3URL}/{file}
```

`.env` 가 OS 환경변수를 덮어쓴다(`load_dotenv(override=True)`). `.env.example` 참고.

## 실행

### 개발

```bash
.venv/bin/python main.py        # $HOST:$PORT 에서 서빙, 먼저 GPU 모델 warmup (~3초)
```

### 운영 (systemd)

```bash
sudo cp prep_stt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prep_stt
sudo systemctl status prep_stt
```

단일 프로세스로만 띄운다 — `uvicorn --workers` **금지** (워커마다 모델 중복 로드 +
GPU 큐가 깨짐).

## API

Base URL: `http://$HOST:$PORT`

### `GET /` — 헬스체크

```json
{ "message": "hello world", "service": "prep_stt" }
```

### `POST /pre_svc/` — 다운로드 + demux

```bash
curl -X POST http://localhost:8080/pre_svc/ \
  -H 'Content-Type: application/json' \
  -d '{"vid": "6", "file": "6.mp4"}'
```

| 필드 | 설명 |
|---|---|
| `vid` | 영상 id — job 디렉토리명 겸용 (`output/{vid}/`) |
| `file` | S3 객체 키 — `{S3URL}/{file}` 로 받음 (확장자가 `.mp4` 아닐 수 있음) |

```json
{
  "status": "ok",
  "job_id": "6",
  "video_path": "output/6/source.mp4",
  "audio_path": "output/6/audio.wav",
  "num_chunks": 42
}
```

### `POST /stt_svc/` — STT 실행 (동기)

```bash
curl -X POST http://localhost:8080/stt_svc/ \
  -H 'Content-Type: application/json' \
  -d '{"vid": "6", "file_path": "output/6/audio.wav"}'
```

| 필드 | 설명 |
|---|---|
| `vid` | job id, `pre_svc` 와 공유 (`output/{vid}/`) |
| `file_path` | 입력 오디오 (NVMe 경로, 예: `pre_svc` 의 `audio.wav`) |

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

`status` 는 `"done"` 또는 `"error"`(메시지는 `error`). 호출은 파이프라인이 끝날 때까지
**블로킹**된다 — RTF ≈ 0.042 라 1시간 입력 ≈ 150초, 6시간 ≈ 15분.
전체 결과는 `output/{vid}/result.json` 에도 기록된다.

> HTTP 계층은 현재 동기식이다. 아주 긴 입력을 위한 비동기 `202 + 폴링` 변형은
> 향후 과제로 둔다.

## 출력

NVMe scratch, job 단위 디렉토리 (durable 사본은 S3 에 둔다):

```
output/{vid}/
├── source.mp4      # 다운로드한 원본
├── audio.wav       # 추출된 mono 16 kHz
├── denoise.wav     # (SAVE_STEPS["denoise"] = True 일 때만)
└── result.json     # 최종 세그먼트
```
