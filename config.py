import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

INTERVIEW_QUESTION_NUMBER = 5

#================================= 모의 면접 평가 부분 설정 ===================================

EVAL_GOODORBAD = 3 #default(3)  강점과 개선점 개수
EVAL_LIMITTIME = 5.0 #default(5)  제한시간
EVAL_IMPROVMENT = 3 #default(3) 개선점

#================================= 자소서 분석 부분 설정 ===================================

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI")

# 자소서 유사도 분석 임계값
# 0.85: 매우 엄격 (거의 동일한 문장만 그룹핑)
# 0.80: 엄격 (의미가 매우 유사한 문장들 그룹핑)
# 0.75: 중간 (의미가 유사한 문장들 그룹핑)
# 0.70: 관대 (어느 정도 관련 있는 문장들도 그룹핑)
SIMILARITY_THRESHOLD = 0.85  # 임시로 높여서 테스트

# 사용할 Sentence Transformer 모델
SENTENCE_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# GPT 모델 설정
GPT_MODEL = "gpt-4o"
GPT_TEMPERATURE = 0.5


