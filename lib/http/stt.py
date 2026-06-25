"""stt_svc 핸들러 (HTTP 라우터) + 요청/응답 DTO. 동기 실행 (결과 직접 반환).

요청 DTO 파싱 → service(plain 인자) 호출 → 응답 DTO 매핑.
"""
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from lib.service import stt_service
from lib.http.http_util import log_req, log_res

router = APIRouter(prefix="/stt_svc", tags=["stt_svc"])


class SttRequest(BaseModel):
    vid: str           # 영상 id — job_id 겸 output/{vid}/ 디렉토리 (pre_svc 와 공유)
    file_path: str     # 입력 오디오 (NVMe 경로, pre 의 audio.wav). 내부서 denoise/VAD/LID/ASR/화자 처리


class SttResponse(BaseModel):
    job_id: str
    status: Literal["done", "error"]
    segments: list[dict] = []    # done 일 때 result.json 내용 그대로 (pass-through). 그 외 []
    error: str = ""              # error 일 때 메시지. 그 외 ""


@router.post("/", response_model=SttResponse)
def submit(req: SttRequest):
    # 동기: STT 파이프라인을 끝까지 돌리고 결과(segments)를 바로 반환 (블로킹)
    log_req(req)
    result = stt_service.submit(req.vid, req.file_path)
    resp = SttResponse(**result)
    return resp
