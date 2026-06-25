"""ffmpeg — 영상 1회 디코드로 두 산출물 생성 (한 명령).

job 디렉토리(output/{vid}/) 안의 source.* 를 입력으로, 같은 디렉토리에:
  audio.wav                 전체 오디오 (mono 16k) — 소리 분석(stt)용
  chunk_00000_00005.mp4 …   6초 영상 청크 (1024x768, 원본fps, 소리포함, H.264/AAC) — 영상 분석용
                            이름 = chunk_{시작초}_{끝초}(5자리). 마지막은 나머지 길이.
CPU/GPU 는 config.FFMPEG_MODE 로 전환(_build_cmd). 입력 타입 검증은 앞단(upload 서버) 가정.
"""
import math
import subprocess
from pathlib import Path

import logging
import config

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT = 3600   # s — 대용량 원본 대비 넉넉히


def extract_and_chunk(job_dir: Path) -> tuple[Path, list[Path]]:
    """job_dir/source.* → audio.wav + chunkNNN.mp4 (같은 디렉토리).

    return: (audio_path, [chunk_path, ...])
    """
    job_dir = Path(job_dir)
    src = next(job_dir.glob("source.*"))            # 다운로드 원본
    audio_out = job_dir / "audio.wav"
    pattern = job_dir / "seg%05d.mp4"               # 임시 순번 → 아래서 시간범위 이름으로 rename

    _run(_build_cmd(src, audio_out, pattern))

    # seg00000.mp4 … → chunk_{시작초}_{끝초}.mp4  (6초 그리드: 시작=인덱스×6, 끝=시작+5)
    total = math.ceil(_duration(src))               # 마지막 청크 끝초 계산용 (전체 길이)
    segs = sorted(job_dir.glob("seg*.mp4"))
    n = len(segs)
    chunks = []
    for i, seg in enumerate(segs):
        start = i * 6
        end = (start + 5) if i < n - 1 else (total - 1)   # 마지막만 나머지
        dst = job_dir / f"chunk_{start:05d}_{end:05d}.mp4"
        seg.rename(dst)
        chunks.append(dst)

    log.info(f"audio → {audio_out} + {n} chunks ({total}s total) → {job_dir}")
    return audio_out, chunks


def _build_cmd(src: Path, audio_out: Path, pattern: Path) -> str:
    """config.FFMPEG_MODE 에 따라 CPU/GPU ffmpeg 명령 생성.

    공통: 출력1=전체 오디오(mono 16k wav, 소리분석) + 출력2=6초 영상청크(1024x768, 원본fps, 소리포함, 영상분석).
    ⚠ fps 는 줄이지 않는다(원본 유지): 1fps 로 줄이면 segment 먹서가 오디오를 6초 경계에 못 맞춰
      청크 오디오가 겹침/잘림. 원본 fps 면 영상·오디오 둘 다 6초로 깔끔히 잘림.
    """
    bin_ = config.FFMPEG_DIR / "ffmpeg"

    if config.FFMPEG_MODE == "gpu":
        # GPU: NVDEC 디코드 + scale_cuda + NVENC. 6초 키프레임은 force_key_frames + -forced-idr 1
        #   (nvenc 는 forced-idr 없으면 force_key_frames 무시). STT 와 다른 GPU 로 핀(-hwaccel_device).
        #   ⚠ -c:v vp9_cuvid 는 입력이 VP9 일 때만.
        return (
            f"{bin_} -y -hwaccel cuda -hwaccel_output_format cuda "
            f"-hwaccel_device {config.FFMPEG_GPU} -c:v vp9_cuvid -i {src} "
            f"-map 0:a -vn -ac 1 -ar {config.TARGET_SR} -c:a pcm_s16le {audio_out} "
            f"-map 0:v -map 0:a -vf scale_cuda=1024:768 -c:v h264_nvenc "
            f"-force_key_frames 'expr:gte(t,n_forced*6)' -forced-idr 1 -c:a aac "
            f"-f segment -segment_time 6 "
            f"-segment_start_number 0 -reset_timestamps 1 -segment_format mp4 {pattern}"
        )

    # CPU: libx264. 디코드 `-threads 1`(입력 옵션) — 멀티스레드 VP9 디코드는 이 콘텐츠에서 race 로
    #   세그폴트. 인코드는 `-threads N`(출력)으로 멀티 OK(libx264 스레딩은 견고). 디코드만 single 이면 안전.
    return (
        f"{bin_} -y -threads 1 -i {src} "
        f"-map 0:a -vn -ac 1 -ar {config.TARGET_SR} -c:a pcm_s16le {audio_out} "
        f"-map 0 -vf scale=1024:768 "
        f"-c:v libx264 -pix_fmt yuv420p -threads {config.FFMPEG_ENC_THREADS} -c:a aac "
        f"-force_key_frames 'expr:gte(t,n_forced*6)' "
        f"-f segment -segment_time 6 "
        f"-segment_start_number 0 -reset_timestamps 1 -segment_format mp4 {pattern}"
    )


def _duration(path: Path) -> float:
    """ffprobe 로 미디어 전체 길이(초) 조회. ffprobe 는 FFMPEG_DIR 에서 찾음(PATH 비의존)."""
    ffprobe = str(config.FFMPEG_DIR / "ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({proc.returncode}): {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def _run(cmd: str) -> None:
    log.info(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
