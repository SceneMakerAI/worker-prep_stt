"""Whisper — faster-whisper (large-v3, non-turbo). LID + ASR 단일 모델.

LID(detect_language)와 ASR(transcribe)은 동일한 WhisperModel 인스턴스의 메서드라
모델을 한 번만 로드해 공유한다 (lid.py 분리 ❌).

이 모듈은 모델 래퍼(primitive)만 제공:
    load_model()                              — 시작 시 1회 (상주)
    detect_language(chunk) -> (lang, prob)    — LID
    transcribe_batched(audio, language) -> [segments]  — 배치 ASR (30s 윈도우 병렬)

라우팅 / 묵음 스트림 구성 / 후처리 필터 / 병합 등 오케스트레이션은 services/stt_service.py 담당.
모델/임계값: config.py, PoC: poc-stt-bench/lib/audio/whisper/whisper_stt.py 참조.
"""
from typing import Optional

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

import config
import logging

log = logging.getLogger(__name__)

_model: Optional[WhisperModel] = None              # LID + ASR 공유
_batched: Optional[BatchedInferencePipeline] = None  # 배치 transcribe 래퍼
BATCH_SIZE = 16   # 동시에 GPU 에 올리는 30s 윈도우 수


def load_model() -> None:
    """시작 시 1회 호출. 로컬 경로에서 로드 (환경변수 비의존)."""
    global _model, _batched
    if _model is not None:
        return
    log.info(f"Loading Whisper from {config.WHISPER['MODEL']}...")
    _model = WhisperModel(
        str(config.WHISPER["MODEL"]),
        device="cuda",
        device_index=config.WHISPER["GPU_NUM"],
        compute_type=config.WHISPER["COMPUTE_TYPE"],
    )
    _batched = BatchedInferencePipeline(model=_model)
    log.info("Whisper ready")


def detect_language(chunk: np.ndarray) -> tuple[str, float]:
    """raw chunk → (lang_code, prob). LID 는 raw 오디오에서 (PoC Strategy 2)."""
    if _model is None:
        load_model()
    lang_code, prob, _all_probs = _model.detect_language(chunk)
    return lang_code, prob


def transcribe_batched(audio: np.ndarray, language: str) -> list[dict]:
    """오디오를 지정 언어로 배치 transcribe (내부 VAD 로 발화 구간만, 30s 윈도우 병렬).

    audio 는 보통 "해당 언어 외 구간을 묵음 처리한 전체 길이 스트림" → 내부 VAD 가
    묵음을 건너뛰므로 그 언어 발화만 인식되고, timestamp 는 원본 시각 그대로.

    word_timestamps 로 단어 단위 반환 → 호출측(stt_service)이 화자 턴/문장부호 기준으로
    재분할 (배치는 segment 가 30s 로 뭉치므로 세밀도/화자 정확도 복원).

    Returns: [{"start", "end", "word", "seg_logprob"}, ...]  (단어 단위, 절대 시각)
             seg_logprob = 그 단어가 속한 segment 의 avg_logprob (환각 필터용)
    """
    if _model is None:
        load_model()
    segments_gen, _info = _batched.transcribe(
        audio,
        language=language,
        batch_size=BATCH_SIZE,
        beam_size=5,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        vad_filter=True,        # 묵음 구간 건너뛰기 (묵음 스트림 처리의 핵심)
        word_timestamps=True,   # 단어 단위 타임스탬프
    )
    words = []
    for s in segments_gen:
        for w in (s.words or []):
            words.append({
                "start": float(w.start), "end": float(w.end),
                "word": w.word, "seg_logprob": s.avg_logprob,
            })
    return words
