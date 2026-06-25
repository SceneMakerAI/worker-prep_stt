"""잡동사니 유틸 — 어디 두기 애매한 작은 헬퍼들."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import logging

log = logging.getLogger(__name__)


def save_wav(path, audio_np: np.ndarray, sr: int) -> None:
    """wav 저장 (16-bit PCM). 부모 디렉토리 자동 생성."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), audio_np, sr, subtype="PCM_16")
    log.info(f"saved wav → {p}")


def write_json(path, obj) -> None:
    """dict → JSON 파일 (utf-8, indent). 부모 디렉토리 자동 생성."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"wrote json → {p}")


def write_text(path, text: str) -> None:
    """텍스트 파일 저장 (utf-8). 부모 디렉토리 자동 생성."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    log.info(f"wrote text → {p}")
