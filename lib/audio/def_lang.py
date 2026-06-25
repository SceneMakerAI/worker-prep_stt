"""Whisper LID 게이트용 언어 정책.

faster-whisper large-v3 의 언어별 WER 기준 티어링.
사용: if lid_lang not in ALLOWED_LANGS: skip
PoC: poc-stt-bench/lib/audio/whisper/def_lang.py
"""

# Tier 1 — 매우 정확 (WER < 5%). 거의 환각 없음.
TIER1_LANGS: frozenset[str] = frozenset({
    "en", "es", "it", "fr", "de", "pt",
})

# Tier 2 — 양호 (WER 5-10%). 주 커버리지 (ko 포함). 후처리 필요하나 신뢰 가능.
TIER2_LANGS: frozenset[str] = frozenset({
    "ko", "ja", "zh", "ru", "nl", "pl", "tr", "ca", "uk",
})

# Tier 3 — 보통 (WER 10-20%). 조심해서 수용 — 짧은 발화는 환각 가능.
TIER3_LANGS: frozenset[str] = frozenset({
    "ar", "he", "hi", "id", "ms", "vi", "el", "hu", "cs", "fi", "sv", "da", "no",
})

# 허용 = Tier 1 ∪ 2 ∪ 3. 이 밖(Tier 4 약함, Tier 5 미지원)은 skip.
ALLOWED_LANGS: frozenset[str] = TIER1_LANGS | TIER2_LANGS | TIER3_LANGS
