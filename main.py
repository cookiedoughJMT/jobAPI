from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.interview_api import interview_api

app = FastAPI(
    title="JobAyong API",
    description="JobAyong 서비스의 백엔드 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(interview_api, prefix="/api/interview", tags=["면접"])
# app.include_router(resume.router, prefix="/api/resume", tags=["이력서"])
# app.include_router(voice.router, prefix="/api/voice", tags=["음성"])

@app.get("/")
async def root():
    return {"message": "JobAyong API 서버가 실행 중입니다."} 