"""VAD — Silero VAD. 발화 구간(speech range) 추출.

torch.hub 기반 (별도 pip 설치 불필요). 최초 호출 시 ~/.cache/torch/hub/ 에 캐시.

발화 단위로 자르면 고정 30s 분할과 달리 경계 클리핑이 없음 → LID/ASR 정확도 ↑.
(PoC: poc-stt-bench/lib/audio/vad.py)

API:
    load_model()                              — 시작 시 1회 (warmup)
    detect(audio_np, sr) -> [(start_s, end_s)]  — 발화 구간 (초)
"""
from typing import Optional

import numpy as np
import torch

import config
import logging

log = logging.getLogger(__name__)

_model: Optional[torch.nn.Module] = None
_get_speech_timestamps = None

# 발화 길이 제약 — LID 정확도와 후속 처리 부하 균형
MIN_SPEECH_S = 1.0     # 너무 짧으면 LID 부정확 ("Yeah" 같은 단음절 환각)
MAX_SPEECH_S = 30.0    # 너무 길면 처리 부하 (Whisper 30s 윈도우 한계)
MIN_SILENCE_S = 0.3    # 이보다 짧은 침묵은 무시 (인접 발화 병합)


def load_model() -> None:
    """시작 시 1회 호출. 로컬 repo 폴더에서 로드 (환경변수/네트워크 비의존)."""
    global _model, _get_speech_timestamps
    if _model is not None:
        return
    log.info(f"Loading Silero VAD from {config.SILERO_VAD_REPO}...")
    _model, utils = torch.hub.load(
        repo_or_dir=str(config.SILERO_VAD_REPO),
        model="silero_vad",
        source="local",
    )
    _get_speech_timestamps = utils[0]
    log.info("Silero VAD ready")


def detect(audio_np: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]:
    """발화 구간 타임스탬프 추출.

    audio_np : float32 1D numpy (16kHz 권장)
    sr       : 샘플레이트 (Silero VAD 는 16k / 8k 만 공식 지원)

    Returns: [(start_sec, end_sec), ...]  — 발화 구간 (정렬됨)
    """
    if _model is None:
        load_model()
    audio_t = torch.from_numpy(audio_np).float()
    ts_list = _get_speech_timestamps(
        audio_t, _model, sampling_rate=sr,
        min_speech_duration_ms=int(MIN_SPEECH_S * 1000),
        max_speech_duration_s=MAX_SPEECH_S,
        min_silence_duration_ms=int(MIN_SILENCE_S * 1000),
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts_list]
