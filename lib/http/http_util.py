"""HTTP 미들웨어 — 요청/응답 로깅.

요청 들어올 때 method+URL, 끝날 때 상태코드+URL+소요시간을 prep_stt.log 에 남긴다.
main.py 에서 register(app) 로 등록.
"""
import logging
import time

from fastapi import Request


def register(app):
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logging.info(f"→ {request.method} {request.url}")
        t0 = time.time()
        response = await call_next(request)
        logging.info(f"← {request.method} {request.url} {response.status_code} ({time.time() - t0:.1f}s)")
        return response


def log_req(req):
    """요청 body 로깅. pydantic 모델은 repr 에 클래스명+필드가 들어감."""
    logging.info(f"req  {req!r}")


def log_res(res):
    """응답 body 로깅."""
    logging.info(f"res  {res!r}")
