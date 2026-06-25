"""denoise — DeepFilterNet v3 음성 향상 (잡음 제거).

BGM/효과음/군중 잡음 제거 → ASR 환각 감소 (Strategy 2: ASR 입력은 denoised).
atten_lim_db=-30 (강도 완화 — 노래/정상 발화 보존). DF native sr=48kHz, 출력도 48k
(downstream 에서 16k 재샘플). 긴 오디오는 30s 청크로 처리 (DF 스펙트로그램 VRAM OOM 방지).

⚠ DeepFilterNet 은 cuda:0 고정 (init_df 가 자동 선택, 다른 디바이스로 옮기면 내부 버퍼
불일치). denoise 는 numpy in/out 이라 whisper(cuda:1) 와 텐서를 안 섞어 무관.

API (메모리 in/out — stt_service.run() 안에서 호출):
    load_model()                  — 시작 시 1회 (상주)
    process(audio_np, sr) -> (np48k, 48000)   — mono 오디오 → denoised 48k float32

PoC: poc-stt-bench/lib/audio/denoise.py
"""
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

# df.io 가 import 하는 torchaudio.AudioMetaData 는 torchaudio 2.9+ 에서 제거됨.
# df import 전에 더미로 shim (실제로는 df.io 대신 soundfile 로 입출력하므로 타입만 필요).
if not hasattr(torchaudio, "AudioMetaData"):
    torchaudio.AudioMetaData = type("AudioMetaData", (), {})

from df.enhance import enhance, init_df  # noqa: E402  (shim 이후에 import 해야 함)

import config  # noqa: E402
import logging  # noqa: E402

log = logging.getLogger(__name__)

_model: Optional[torch.nn.Module] = None
_df_state = None

ATTEN_LIM_DB = -30   # 잡음 감쇠 상한(dB). None=full power, -30=음성 보존
CHUNK_SEC = 30       # 긴 오디오 청크 길이 (DF 스펙트로그램 VRAM 한계)


def load_model() -> None:
    """시작 시 1회. 로컬 경로에서 오프라인 로드 (cuda:0 고정)."""
    global _model, _df_state
    if _model is not None:
        return
    log.info(f"Loading DeepFilterNet3 from {config.DEEPFILTER_MODEL}...")
    _model, _df_state, _, _ = init_df(   # 4-tuple (model, df_state, suffix, epoch)
        model_base_dir=str(config.DEEPFILTER_MODEL),
        log_file=None,
    )
    log.info(f"DeepFilterNet3 ready (sr={_df_state.sr()}, "
             f"device={next(_model.parameters()).device})")


def process(audio_np: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """mono 오디오 → denoised. 반환: (denoised_48k float32, 48000).

    원본 sr 그대로 받아 DF native(48k)로 resample 후 처리 (품질 보존 — 16k 다운샘플은
    호출측에서 denoise 후에). 긴 오디오(>CHUNK_SEC)는 청크 처리 후 concat (GPU OOM 방지).
    """
    if _model is None:
        load_model()
    df_sr = _df_state.sr()

    audio = torch.from_numpy(audio_np)
    if audio.ndim > 1:
        audio = audio.mean(dim=1)        # → mono
    audio = audio.unsqueeze(0)           # [1, T]
    if sr != df_sr:
        audio = F.resample(audio, sr, df_sr)

    total = audio.shape[-1]
    chunk = CHUNK_SEC * df_sr
    if total <= chunk:
        enhanced = enhance(model=_model, df_state=_df_state, audio=audio,
                           pad=True, atten_lim_db=ATTEN_LIM_DB)
    else:
        n = (total + chunk - 1) // chunk
        log.info(f"denoise chunked: {total / df_sr:.1f}s → {n} chunks of {CHUNK_SEC}s")
        parts = []
        for s in range(0, total, chunk):
            parts.append(enhance(
                model=_model, df_state=_df_state, audio=audio[..., s:s + chunk],
                pad=True, atten_lim_db=ATTEN_LIM_DB,
            ))
        enhanced = torch.cat(parts, dim=-1)

    return enhanced.squeeze(0).cpu().numpy().astype(np.float32), df_sr
