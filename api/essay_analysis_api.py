from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from service.essay_analysis_service import EssayAnalysisService

logger = logging.getLogger(__name__)

essay_analysis_api = APIRouter()

# 요청 모델
class EssayAnalysisRequest(BaseModel):
    original_sentences: List[str] = Field(
        ..., 
        description="완전 중복 제거된 자소서 문장 배열",
        example=[
            "회사의 발전에 기여하는 동시에, 저 역시 지속적으로 성장하는 개발자가 되겠습니다.",
            "항상 열린 마음으로 배우고, 팀과 함께 성장하는 자세를 잃지 않겠습니다."
        ]
    )

# 응답 모델
class CorrectionItem(BaseModel):
    original: str = Field(description="원본 문장")
    improved: str = Field(description="개선된 문장")
    reason: str = Field(description="개선 이유")

class InsertionPoint(BaseModel):
    target_sentence: str = Field(description="삽입할 대상 문장")
    reason: str = Field(description="삽입 이유")

class SuggestionItem(BaseModel):
    title: str = Field(description="제안 제목")
    content: str = Field(description="제안 내용")
    example: str = Field(description="예시 문장")
    insertion_point: InsertionPoint = Field(description="삽입 위치 정보")

class MissingArea(BaseModel):
    category: str = Field(description="부족한 영역 카테고리")
    description: str = Field(description="영역 설명")
    suggestions: List[SuggestionItem] = Field(description="개선 제안들")

class AnalysisResult(BaseModel):
    corrections: List[CorrectionItem] = Field(description="문법/표현 교정 및 중복 제거 사항")
    missing_areas: List[MissingArea] = Field(description="부족한 영역 및 제안")

class GroupedData(BaseModel):
    group_id: int = Field(description="그룹 ID")
    sentences: List[str] = Field(description="그룹 내 문장들")
    sentence_indices: List[int] = Field(description="원본 문장 인덱스들")
    is_similar_group: bool = Field(description="유사 문장 그룹 여부")

class EssayAnalysisResponse(BaseModel):
    original_sentences_count: int = Field(description="원본 문장 수")
    groups: List[List[int]] = Field(description="유사 문장 그룹 인덱스")
    grouped_data: List[GroupedData] = Field(description="그룹화된 문장 데이터")
    analysis: AnalysisResult = Field(description="GPT 분석 결과")

# 서비스 인스턴스 생성
essay_service = EssayAnalysisService()

@essay_analysis_api.post(
    "/analyze",
    response_model=EssayAnalysisResponse,
    summary="자소서 문장 분석",
    description="""
    자소서 문장들을 분석하여 다음을 수행합니다:
    1. 의미 유사도 기반 문장 그룹핑
    2. 각 그룹의 대표 문장 선택
    3. GPT를 통한 문법/표현 개선 및 부족한 영역 분석
    4. 구조화된 JSON 응답 반환
    """
)
async def analyze_essay(request: EssayAnalysisRequest):
    """자소서 문장 분석 API"""
    try:
        if not request.original_sentences:
            raise HTTPException(
                status_code=400,
                detail="분석할 문장이 제공되지 않았습니다."
            )
        
        # 빈 문장 필터링
        filtered_sentences = [
            sentence.strip() 
            for sentence in request.original_sentences 
            if sentence.strip()
        ]
        
        if not filtered_sentences:
            raise HTTPException(
                status_code=400,
                detail="유효한 문장이 없습니다."
            )
        
        logger.info(f"자소서 분석 시작: {len(filtered_sentences)}개 문장")
        
        # 자소서 분석 수행
        result = essay_service.analyze_essay(filtered_sentences)
        
        logger.info(f"자소서 분석 완료: {len(result.get('analysis', {}).get('corrections', []))}개 수정사항")
        
        return EssayAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"자소서 분석 API 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"자소서 분석 중 오류가 발생했습니다: {str(e)}"
        )

@essay_analysis_api.get(
    "/health",
    summary="자소서 분석 서비스 상태 확인",
    description="자소서 분석 서비스의 상태를 확인합니다."
)
async def health_check():
    """서비스 상태 확인"""
    try:
        # 간단한 테스트 문장으로 서비스 동작 확인
        test_sentences = ["안녕하세요.", "반갑습니다."]
        groups = essay_service.group_similar_sentences(test_sentences)
        
        return {
            "status": "healthy",
            "message": "자소서 분석 서비스가 정상 작동 중입니다.",
            "test_result": f"테스트 그룹핑 완료: {len(groups)}개 그룹"
        }
    except Exception as e:
        logger.error(f"서비스 상태 확인 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서비스 상태 확인 중 오류 발생: {str(e)}"
        ) 