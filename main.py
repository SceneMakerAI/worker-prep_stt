"""
app 생성 + 라우터 등록만. 실제 로직은 lib/ 아래 계층에 둠:
    lib/http          핸들러(라우터) + 요청/응답 DTO   전송 계층 (HTTP)
    lib/service       비즈니스 로직 (transport 무관)
    lib/audio         모델 컴포넌트 (VAD/LID/denoise/whisper, GPU)
    lib/io            S3 / ffmpeg (CPU)
    lib/logging.py    로깅 설정 (basicConfig, 파일 전용). import 시 1회 실행.

실행:
    uv run uvicorn main:app --host 0.0.0.0 --port 8888 --reload
"""
import faulthandler
import logging
import os
import ssl
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

import config

# 네이티브 크래시(segfault) C 레벨 traceback 을 stderr(=journald) 로 출력. (어느 라이브러리인지 특정용)
faulthandler.enable()

from lib.http import pre, stt, http_util
from lib.audio import denoise, speaker, vad, whisper


@asynccontextmanager
async def lifespan(app: FastAPI):
    # OpenSSL 을 메인 스레드에서 선(先)초기화. 이 환경의 OpenSSL 은 비-메인 스레드에서
    # 처음 초기화되면 깨져서(SSLError _ssl.c:3123), sync 핸들러(threadpool)의 requests 가 실패함.
    # 여기서 한 번 만들어두면 이후 워커 스레드의 HTTPS 가 정상 동작.
    ssl.create_default_context()

    # startup — GPU 모델 warmup (상주). 첫 요청 전에 미리 로드.
    logging.info("[warmup] loading GPU models (Denoise + VAD + Whisper + Speaker)...")
    t0 = time.time()
    denoise.load_model()
    vad.load_model()
    whisper.load_model()
    speaker.load_model()
    logging.info(f"[warmup] done ({time.time() - t0:.1f}s)")
    logging.info(f"host={config.HOST}:{config.PORT}")
    yield
    logging.info("System shutdown")
    # shutdown — (필요 시 정리)


app = FastAPI(title="prep_stt", version="0.1.0", lifespan=lifespan)

app.include_router(pre.router)
app.include_router(stt.router)

http_util.register(app)


@app.get("/")
def root():
    return {"message": "hello world", "service": "prep_stt"}


if __name__ == "__main__":
    os.makedirs(config.LOG_DIR, exist_ok=True)

    uvicorn.run(app, host=config.HOST, port=config.PORT)
