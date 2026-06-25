"""pre_svc 비즈니스 로직 — S3 → NVMe → ffmpeg (추출 + 6초 분할).

흐름:
    vid + file → s3.download              output/{vid}/source.*       (원본)
              → ffmpeg.extract_and_chunk  output/{vid}/audio.wav      (전체 오디오, 소리분석)
                                          output/{vid}/chunk001.mp4…  (6초 영상, 영상분석)
CPU/IO bound. GPU 안 씀.

transport 무관 — HTTP DTO 대신 plain 인자/dict 사용 (api 가 DTO 로 매핑).
"""
import config
from lib.io import ffmpeg, s3
import logging

log = logging.getLogger(__name__)


def process(vid: str, file: str) -> dict:
    """vid + S3 파일키 받아 다운로드 → 음성 추출 + 6초 영상청크.

    return: {job_id, video_path, audio_path, num_chunks}
    """
    job_dir = config.JOBS_DIR / vid

    # 1) S3 → NVMe
    video_path = s3.download(vid, file)

    # 2) 한 ffmpeg 명령으로 전체 오디오(소리분석) + 6초 영상청크(영상분석) 동시 생성
    audio_path, chunks = ffmpeg.extract_and_chunk(job_dir)

    return {
        "job_id": vid,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "num_chunks": len(chunks),
    }
