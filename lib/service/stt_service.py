"""stt_svc 비즈니스 로직 — STT 파이프라인 오케스트레이션 (GPU).

파이프라인 (PoC Strategy 2 — VAD/LID=raw, ASR=denoised):
    0. 오디오 로드     → raw(16k) + denoised(16k)
    1. VAD(raw)        → 발화 구간 [(start,end)...]
    2. LID + 라우팅    → 구간별 언어 판정, 언어별 그룹 (짧고 비-주언어면 dual)   [GPU+CPU]
    3. transcribe      → 언어별 transcribe (추후 배치)                          [GPU]
    4. 병합 + 필터     → dual logprob 비교, 후처리 게이트, timestamp 정렬       [CPU]
    5. result.json 저장 (NVMe)

동기: submit() 이 요청에서 run() 을 직접 실행하고 결과(segments)를 반환 (블로킹).
GPU 는 _run_lock 으로 한 번에 한 job 만 (직렬화).

transport 무관 — HTTP DTO 대신 plain 인자/dict 사용 (api 가 DTO 로 매핑).
세부 임계값: config.py / PoC: poc-stt-bench/lib/audio/whisper/whisper_stt.py
"""
import gc
import json
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

import config
from lib import util
from lib.audio import denoise, speaker, vad, whisper
from lib.audio.def_lang import ALLOWED_LANGS
import logging

log = logging.getLogger(__name__)

# ── 라우팅/후처리 튜닝값 (알고리즘 상수 — PoC whisper_stt 검증값) ──────────
MAIN_LANG = "ko"          # 주 언어 (LID/dual 기준)
LID_TRUST_PROB = 0.5      # LID prob 이 이 미만 + 비-주언어 → 주언어 강제
SHORT_SEG_S = 3.0         # 이 미만 + 비-주언어 → dual (ko/lid 비교)
MIN_LOGPROB = -1.0        # segment avg_logprob 이 이 미만 → 환각으로 drop


# ─────────────────────────────────────────────────────────────
# 동기 진입점 (api 가 호출) — 요청에서 직접 실행, GPU 는 lock 으로 직렬화
# ─────────────────────────────────────────────────────────────
_run_lock = threading.Lock()   # GPU 는 한 번에 한 job — 동기 요청 직렬화


def submit(vid: str, file_path: str) -> dict:
    """동기 실행 — 요청에서 STT 파이프라인을 직접 돌려 결과(segments)를 반환. (블로킹)

    vid       : job_id 겸 output/{vid}/ 디렉토리 (pre_svc 와 공유). result.json 도 여기 저장.
    file_path : 입력 오디오 (mono wav, pre 의 audio.wav). denoise/VAD/LID/ASR/화자 구분 전부 내부 처리.
    GPU 는 _run_lock 으로 직렬화 (동시 요청이 와도 한 번에 하나만 GPU 사용).
    """
    job_id = vid
    out_path = str(config.JOBS_DIR / vid / "result.json")
    with _run_lock:
        try:
            run(file_path, out_path)
            status_, error = "done", ""
        except Exception as e:  # noqa: BLE001 — 실패도 응답으로 전달
            log.exception(f"job {job_id} failed")
            status_, error = "error", str(e)
        finally:
            # GPU 메모리 누적 방지 (대용량 파일 연속 처리 시 OOM/segfault 완화)
            gc.collect()
            torch.cuda.empty_cache()

    segments = []
    if status_ == "done":
        segments = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return {"job_id": job_id, "status": status_, "segments": segments, "error": error}


# ─────────────────────────────────────────────────────────────
# 핵심 파이프라인 (백그라운드 워커가 실행)
# ─────────────────────────────────────────────────────────────
def run(audio_path: str, out_path: str) -> dict:
    """STT 파이프라인 전체 실행 → result.json 저장, 요약 반환."""
    # 0) 오디오 로드 — raw(VAD/LID용) + denoised(ASR용, 내부 denoise)
    raw, den = _load_audio(audio_path, job_dir=Path(out_path).parent)

    # 1) VAD — raw 에서 발화 구간
    ranges = vad.detect(raw, sr=config.TARGET_SR)

    # 1b) 화자 구분 — raw 전체에서 화자 턴 타임라인 (1회)
    turns = speaker.diarize(raw, sr=config.TARGET_SR)

    # 2) LID 분류 — 구간별 언어 판정 → { lang: [(start,end), ...] }
    groups = _classify_languages(raw, ranges)

    # 3) 언어별 묵음 스트림 배치 transcribe (denoised) → 단어 단위 (절대 시각)
    words = _transcribe_batched(den, groups)

    # 4) 단어 → segment 재조립 (필터 + 화자 매핑 + 화자/문장 경계 분할)
    segments = _assemble_segments(words, turns)

    # 5) 저장 = result.json — idx + 시각(HH:MM:SS.s) + speaker(S025)
    seg_json = [
        {"idx": i,
         "start": _fmt_time(s["start"]), "end": _fmt_time(s["end"]),
         "text": s["text"], "lang": s["lang"], "speaker": _fmt_speaker(s["speaker"])}
        for i, s in enumerate(segments)
    ]
    util.write_json(out_path, seg_json)

    log.info(f"run done: {len(segments)} segments → {out_path}")
    return {"out_path": out_path, "num_segments": len(segments)}


def _fmt_time(t: float) -> str:
    """초 → HH:MM:SS.s (예: 296.8 → 00:04:56.8)."""
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _fmt_speaker(label) -> str:
    """pyannote 라벨 → Snnn (예: SPEAKER_25 → S025). None 은 'None'(4자, Snnn 과 폭 동일)."""
    if not label:
        return "None"
    num = label.rsplit("_", 1)[-1]
    return f"S{int(num):03d}" if num.isdigit() else str(label)


# ─────────────────────────────────────────────────────────────
# 파이프라인 단계 헬퍼 (추후 구현)
# ─────────────────────────────────────────────────────────────
def _load_audio(audio_path: str, job_dir: Path = None):
    """입력 wav → (raw_np, den_np). 둘 다 16k mono float32, 길이 정렬.

    Strategy 2: VAD/LID/화자 구분 = raw, ASR = denoised.
    품질 보존을 위해 원본 sr 에 denoise 를 걸고(48k), 그 뒤 16k 로 다운샘플.
    config.SAVE_STEPS["denoise"] 가 켜지면 denoised(48k)를 job 폴더에 저장(검수용).
    """
    orig, osr = sf.read(str(audio_path), dtype="float32")
    if orig.ndim > 1:
        orig = orig.mean(axis=1)                       # → mono

    raw_np = _to_16k(orig, osr)                         # VAD/LID/speaker 용
    den48, dsr = denoise.process(orig, osr)            # 원본에 denoise → 48k

    if config.SAVE_STEPS.get("denoise") and job_dir is not None:
        util.save_wav(job_dir / "denoise.wav", den48, dsr)

    den_np = _to_16k(den48, dsr)                        # ASR 용 16k

    n = min(len(raw_np), len(den_np))                  # 길이 차 안전 처리
    log.info(f"audio loaded: {n / config.TARGET_SR:.1f}s (raw + internal-denoise, 16k mono)")
    return raw_np[:n], den_np[:n]


def _to_16k(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """mono float32 numpy → 16k."""
    if sr == config.TARGET_SR:
        return audio_np.astype(np.float32)
    t = F.resample(torch.from_numpy(audio_np), sr, config.TARGET_SR)
    return t.numpy().astype(np.float32)


def _classify_languages(raw, ranges):
    """각 구간 LID(raw) → 언어별 시간범위 그룹 { lang: [(start_s, end_s), ...] }.

    ALLOWED_LANGS 게이트 + LID_TRUST_PROB(비-주언어 저신뢰 → 주언어 강제) 적용.
    (배치 방식이라 PoC 의 dual 비교는 생략 — LID_TRUST 게이트로 갈음)
    """
    sr = config.TARGET_SR
    main = MAIN_LANG
    groups: dict[str, list] = {}
    for start_s, end_s in ranges:
        chunk = raw[int(start_s * sr):int(end_s * sr)]
        lang, prob = whisper.detect_language(chunk)

        if lang not in ALLOWED_LANGS:                       # Tier 4-5 버림
            log.info(f"  [{start_s:.1f}-{end_s:.1f}] LID {lang}={prob:.2f} → not allowed, skip")
            continue
        if lang != main and prob < LID_TRUST_PROB:          # 저신뢰 → 주언어
            lang = main

        groups.setdefault(lang, []).append((start_s, end_s))

    log.info(f"languages: { {k: len(v) for k, v in groups.items()} }")
    return groups


def _transcribe_batched(den, groups):
    """언어별로 '그 언어 외 구간을 묵음 처리한 전체 길이 스트림'을 만들어 배치 transcribe.

    내부 VAD 가 묵음을 건너뛰므로 그 언어 발화만 인식 + timestamp 는 원본 시각 그대로.
    반환(flat 단어): [{"start", "end", "word", "seg_logprob", "lang"}, ...]  (절대 시각)
    """
    sr = config.TARGET_SR
    out = []
    for lang, lang_ranges in groups.items():
        stream = np.zeros_like(den)                          # 전체 묵음
        for start_s, end_s in lang_ranges:                  # 해당 언어 구간만 복원
            i, j = int(start_s * sr), int(end_s * sr)
            stream[i:j] = den[i:j]
        words = whisper.transcribe_batched(stream, language=lang)
        for w in words:
            w["lang"] = lang
        out.extend(words)
        log.info(f"batched [{lang}]: {len(words)} words from {len(lang_ranges)} ranges")
    return out


def _assemble_segments(words, turns):
    """단어 → segment 재조립. 환각 필터 + 화자 매핑 + 화자/언어/문장/간격 경계 분할.

    배치는 segment 가 30s 로 뭉치므로, 단어 단위로 받아 여기서 fine segment 를 복원하고
    화자 정확도를 살린다.
    반환: [{"start","end","text","lang","speaker"}, ...]  (start 정렬)
    """
    SENT_END = (".", "?", "!", "。", "？", "！")
    MAX_GAP_S = 1.0   # 같은 화자라도 이 이상 벌어지면 끊음

    # 1) 환각 게이트 + 빈 단어 제거 + 시간순 정렬
    words = [w for w in words if w["seg_logprob"] >= MIN_LOGPROB and w["word"].strip()]
    words.sort(key=lambda w: w["start"])

    # 2) 단어별 화자 매핑
    for w in words:
        w["speaker"] = _speaker_of(w["start"], w["end"], turns)

    # 3) 그룹핑 — 화자/언어 바뀌거나, 큰 간격, 직전 단어가 문장 끝이면 분할
    segments = []
    cur = []

    def _flush():
        if not cur:
            return
        segments.append({
            "start": round(cur[0]["start"], 2),
            "end": round(cur[-1]["end"], 2),
            "text": "".join(w["word"] for w in cur).strip(),
            "lang": cur[0]["lang"],
            "speaker": cur[0]["speaker"],
        })
        cur.clear()

    for w in words:
        if cur and (w["speaker"] != cur[-1]["speaker"]
                    or w["lang"] != cur[-1]["lang"]
                    or w["start"] - cur[-1]["end"] > MAX_GAP_S):
            _flush()
        cur.append(w)
        if w["word"].strip().endswith(SENT_END):
            _flush()
    _flush()

    log.info(f"assemble: {len(segments)} segments from {len(words)} words")
    return segments


def _speaker_of(start, end, turns):
    """[start,end] 와 시간 겹침이 가장 큰 화자 턴 → speaker label."""
    best_spk, best_overlap = None, 0.0
    for ts, te, spk in turns:
        overlap = min(end, te) - max(start, ts)
        if overlap > best_overlap:
            best_overlap, best_spk = overlap, spk
    return best_spk
