"""pre_svc 핸들러 (HTTP 라우터) + 요청 DTO. 파싱 → service 호출 → 응답."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lib.service import pre_service
from lib.http.http_util import log_req, log_res
import logging

router = APIRouter(prefix="/pre_svc", tags=["pre_svc"])
log = logging.getLogger(__name__)


class PreRequest(BaseModel):
    vid: str           # 영상 id — output/{vid}/ 디렉토리명 (job 단위)
    file: str          # S3 객체 키/파일경로 — S3URL/{file} 로 받음 (확장자 mp4 아닐 수 있음)


@router.post("/")
def pre_svc(req: PreRequest):
    # vid + file 로 S3 원본 다운로드 → ffmpeg (오디오 추출 + 6초 영상청크)
    log_req(req)
    try:
        result = pre_service.process(req.vid, req.file)
    except Exception as e:  # noqa: BLE001 — 처리 실패는 502 로 변환
        log.exception(f"[{req.vid}] pre_svc failed")
        raise HTTPException(status_code=502, detail=f"pre_svc failed: {e}")
    resp = {"status": "ok", **result}
    log_res(resp)
    return resp
