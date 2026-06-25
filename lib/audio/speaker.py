"""speaker — 화자 구분 (Speaker Diarization). pyannote speaker-diarization-community-1.

Whisper 는 화자 식별을 못 하므로 별도 모델. 전체 오디오에서 화자 턴 타임라인을 뽑고,
이후 stt_service 가 각 STT segment 를 시간 겹침(overlap)으로 화자에 매핑한다.

오디오는 메모리상 waveform 으로 직접 넣어 torchcodec(파일 디코딩) 우회.

API:
    load_model()                              — 시작 시 1회 (상주)
    diarize(audio_np, sr) -> [(start_s, end_s, speaker), ...]
"""
import warnings
from typing import Optional

import numpy as np
import torch

# torchcodec(오디오 파일 디코딩용) 로드 실패 경고 억제.
# 화자 구분은 in-memory waveform 직접 전달로 우회하므로 torchcodec 불필요 —
# 미사용 의존성의 긴 traceback 으로 운영 로그가 오염되는 것 방지. (import 전에 필터)
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*", category=UserWarning)

from pyannote.audio import Pipeline  # noqa: E402  (warnings 필터 이후 import)

import config  # noqa: E402
import logging  # noqa: E402

log = logging.getLogger(__name__)

_pipeline: Optional[Pipeline] = None
_device: Optional[torch.device] = None


def load_model() -> None:
    """시작 시 1회. 로컬 config.yaml 에서 로드 (self-contained, 오프라인).

    디바이스는 whisper 와 동일 인덱스로 명시 고정 — bare "cuda" 는 스레드별
    current device 에 의존해 워커 스레드에서 디바이스 불일치(cuda:0 vs cuda:1)를 유발.
    """
    global _pipeline, _device
    if _pipeline is not None:
        return
    _device = torch.device("cuda", config.WHISPER["GPU_NUM"])
    cfg = config.PYANNOTE_DIARIZE / "config.yaml"
    log.info(f"Loading pyannote diarization from {cfg} on {_device}...")
    _pipeline = Pipeline.from_pretrained(str(cfg))
    _pipeline.to(_device)
    log.info("Diarization ready")


def diarize(audio_np: np.ndarray, sr: int = 16000) -> list[tuple[float, float, str]]:
    """화자 턴 타임라인 추출.

    audio_np : float32 1D numpy (16k mono)
    Returns  : [(start_s, end_s, speaker_label), ...]  (시간순)
    """
    if _pipeline is None:
        load_model()
    # (1, num_samples) waveform 으로 직접 전달 → 파일 디코딩 우회.
    # 입력도 모델과 같은 디바이스로 명시 이동 (스레드 간 디바이스 불일치 방지).
    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(_device)
    out = _pipeline({"waveform": waveform, "sample_rate": sr})
    # pyannote 4.x: DiarizeOutput.speaker_diarization 이 Annotation
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in out.speaker_diarization.itertracks(yield_label=True)
    ]
    log.info(f"diarization: {len(turns)} turns, {len(set(t[2] for t in turns))} speakers")
    return turns
