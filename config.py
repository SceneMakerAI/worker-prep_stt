"""서비스 설정 — 경로 / 포트 / 모델 / 임계값.

추후 환경변수(.env) 로딩으로 교체 예정. 지금은 기본값만.
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

load_dotenv(override=True)   # .env 가 기존 OS 환경변수(USER=root 등)를 이기도록 override

# ── 서버
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

# ── 로그
LOG_DIR = "/usr/service/logs/scenemaker"
os.makedirs(LOG_DIR, exist_ok=True)   # basicConfig 가 import 시점에 파일을 열므로 디렉토리 먼저 보장
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s',
    filename=f"{LOG_DIR}/prep_stt.log",
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)


# ── 스토리지
# JOBS_DIR = Path("/mnt/nvme/jobs")   # job 단위 scratch. {job_id}/ 아래 source/audio/result
JOBS_DIR = Path(os.getenv("JOBS_DIR"))
TARGET_SR = 16000                    # mono 16kHz
# 영상 청크 옵션(6초/1024x768/1fps)은 ffmpeg.py 에 하드코딩.
S3URL = os.getenv("S3URL")

# ── ffmpeg 실행 (CPU / GPU 전환).
FFMPEG_MODE = "gpu"                             # "cpu" | "gpu"
FFMPEG_BIN = os.getenv("FFMPEG_BIN")     # 바이너리 경로 (gpu 면 /opt/ffmpeg-nvidia/ffmpeg)
FFMPEG_GPU = os.getenv("FFMPEG_GPU")                     # gpu 모드일 때 사용할 GPU 인덱스 (STT 와 분리)
FFMPEG_ENC_THREADS = 16                         # cpu 모드 인코드 스레드. ⚠ 디코드는 1 고정(멀티면 VP9 디코드 race 로 세그폴트)



# ── 중간 산출물 저장 (디버깅/검수용). job 폴더(output/{job_id}/)에 저장.
SAVE_STEPS = {
    "denoise": False,   # denoised 오디오 → {job_id}/denoise.wav
}

# ── 모델 (PoC 검증값)
# 모델은 환경변수(XDG_CACHE_HOME 등)에 의존하지 않고 코드에서 고정 경로로 로드.
MODELS_ROOT = Path(os.getenv("MODELS_ROOT"))
DEEPFILTER_MODEL = MODELS_ROOT / "deepfilternet3"     # denoise (DeepFilterNet v3, init_df model_base_dir)
SILERO_VAD_REPO = MODELS_ROOT / "silero-vad"          # torch.hub source="local"

WHISPER = {
    "MODEL" : MODELS_ROOT / "whisper-large-v3",
    "GPU_NUM" : int(os.getenv("WHISPER_GPU_NUM")),
    "COMPUTE_TYPE" : "float16",
}

PYANNOTE_DIARIZE = MODELS_ROOT / "pyannote-diarization"  # 화자 구분 (community-1, self-contained)



