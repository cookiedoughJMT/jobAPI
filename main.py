from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from api.interview_api import interview_api
from api.resume_api import resume_api
from api.essay_analysis_api import essay_analysis_api
from api.audio_analysis_api import audio_api

# 파일 크기 제한 미들웨어
class FileSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/ai/audio/analyze" and request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
                max_size = 100 * 1024 * 1024  # 100MB
                if file_size > max_size:
                    return JSONResponse(
                        status_code=413,
                        content={"error": f"파일 크기가 너무 큽니다. ({(file_size / 1024 / 1024):.1f}MB / 100MB)"}
                    )
        return await call_next(request)

app = FastAPI(
    title="JobAyong API",
    description="JobAyong 서비스의 백엔드 API",
    version="1.0.0"
)

# 미들웨어 추가
app.add_middleware(FileSizeMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging

logging.basicConfig(
    level=logging.DEBUG,  # 또는 INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# 라우터 등록
app.include_router(interview_api, prefix="/ai/interview", tags=["면접"])
app.include_router(resume_api, prefix="/ai/resume", tags=["이력서"])
app.include_router(essay_analysis_api, prefix="/ai/essay", tags=["자소서"])
app.include_router(audio_api, prefix="/ai/audio", tags=["음성"])

@app.get("/")
async def root():
    return {"message": "JobAyong API 서버가 실행 중입니다."} 