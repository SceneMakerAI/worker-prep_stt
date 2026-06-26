"""S3 입출력 — durable 저장소(S3) → NVMe scratch.

config.S3URL 기준으로 vid 에 해당하는 원본 영상을 HTTP 로 받아 NVMe 에 저장.
(비공개 버킷이면 presigned/공개 설정이 돼 있어야 받아짐 — 403 이면 권한 문제)
"""
from pathlib import Path

import requests

import config
import logging

log = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 3600     # s — 대용량 원본 대비 넉넉히
CHUNK = 1 << 20             # 1MB 씩 스트리밍 (전체를 메모리에 안 올림)


def download(vid: str, file: str) -> Path:
    """S3URL/{vid%50}/{file} → output/{vid}/{파일명}. 저장된 로컬 경로 반환.

    S3 객체는 vid 를 50 으로 나눈 나머지로 샤딩됨: {S3URL}/{vid%50}/{file}
    (예: vid=5 → vod/5/5.mp4, vid=123 → vod/23/123.mp4). 샤딩은 S3 정리용, 로컬 경로와 무관.
    file : S3 객체 파일명 (예: "6.mp4"). 확장자 무관.
    실패(네트워크/403/404 등) 시 예외 → 호출부에서 실패 처리.
    """
    url = f"{config.S3URL}/{int(vid) % 50}/{file}"
    dest = config.JOBS_DIR / vid / f"source{Path(file).suffix}"   # 고정 이름, 확장자만 원본 유지
    dest.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"[{vid}] download {url} → {dest}")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()                                   # 4xx/5xx → 예외
        with open(dest, "wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)

    size = dest.stat().st_size
    log.info(f"[{vid}] downloaded {size / 1e6:.1f} MB → {dest}")
    return dest
