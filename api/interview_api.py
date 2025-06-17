from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re
from prompt.question_prompt import generate_json_pressure_prompt, generate_json_personality_prompt,generate_json_evaluation, generate_json_technical_prompt, generate_json_situational_prompt,generate_json_general_prompt
from prompt.json_refactor import extract_and_fix_gpt_json, merge_numbered_fields
import config as c

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

interview_api = APIRouter()

class InterviewRequest(BaseModel):
    job_role: Optional[str] = None
    company : Optional[str] = None
    cnt : Optional[int] = 5

class InterviewEvaluationRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    times: list[float]
    limit_time: float = c.EVAL_LIMITTIME
    goodorbad_num: int = c.EVAL_GOODORBAD
    improvment_num: int = c.EVAL_IMPROVMENT
    prev_badpoints: list[str] | None
# ========================================================================== 통합 면접 API==========================================================================================

@interview_api.post("/general_interview")
async def generate_general_interview(request:InterviewRequest):
    prompt, modes = generate_json_general_prompt(request.job_role, request.company, 7)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 반복 없이 다양한 성향의 질문을 구성하는 AI야. 같은 주제를 변형해서 다시 말하지 말고, 질문 간의 명확한 주제 차이를 유지해."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"(\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())

            # 여기에 modes 삽입
            if isinstance(parsed, dict) and "questions" in parsed:
                parsed["modes"] = modes
                return parsed
            else:
                return {"error": "유효한 questions 키가 없습니다.", "raw": content}
        else:
            return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    except Exception as e:
        return {"error": str(e)}

# ========================================================================== 압박 면접 API ==========================================================================================

@interview_api.post("/pressure_interview")
async def generate_pressure_interview(request:InterviewRequest):
    prompt = generate_json_pressure_prompt(request.job_role, request.company, request.cnt)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 반복 없이 다양한 성향의 질문을 구성하는 AI야. 같은 주제를 변형해서 다시 말하지 말고, 질문 간의 명확한 주제 차이를 유지해."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    except Exception as e:
        return {"error": str(e)}

# ========================================================================== 인성 면접 API ==========================================================================================

@interview_api.post("/personality_interview")
async def generate_personality_interview(reqeust:InterviewRequest):

    prompt = generate_json_personality_prompt(reqeust.job_role, reqeust.company)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 반복 없이 다양한 성향의 질문을 구성하는 AI야. 같은 주제를 변형해서 다시 말하지 말고, 질문 간의 명확한 주제 차이를 유지해."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    except Exception as e:
        return {"error": str(e)}

# ========================================================================== 기술심층 면접 API ==========================================================================================

@interview_api.post("/technical_interview")
async def generate_technical_interview(reqeust:InterviewRequest):

    prompt = generate_json_technical_prompt(reqeust.job_role, reqeust.company)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 반복 없이 다양한 성향의 질문을 구성하는 AI야. 같은 주제를 변형해서 다시 말하지 말고, 질문 간의 명확한 주제 차이를 유지해."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    except Exception as e:
        return {"error": str(e)}
# ========================================================================== 상황 면접 평가 API ==========================================================================================

@interview_api.post("/situational_interview")
async def generate_situational_interview(reqeust:InterviewRequest):

    prompt = generate_json_situational_prompt(reqeust.job_role, reqeust.company)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 반복 없이 다양한 성향의 질문을 구성하는 AI야. 같은 주제를 변형해서 다시 말하지 말고, 질문 간의 명확한 주제 차이를 유지해."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    except Exception as e:
        return {"error": str(e)}

# ========================================================================== 모의 면접 평가 API ==========================================================================================

@interview_api.post("/evaluation")
async def generate_evaluation(dto:InterviewEvaluationRequest):
    prompt =  generate_json_evaluation(dto.questions, dto.answers, dto.times, dto.limit_time, dto.goodorbad_num, dto.improvment_num, dto.prev_badpoints)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system",
                 "content": "You are a professional AI that evaluates and provides feedback on interview responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()
        data = extract_and_fix_gpt_json(content)
        data = merge_numbered_fields(data)
        return data

    except Exception as e:
        return {"error": f"API 요청 실패: {str(e)}"}

# ========================================================================== 테스트 API ==========================================================================================

@interview_api.post("/test")
async def generate_test():
    prompt = f"""
    다음 문장에서 핵심을 못잡는 답변인지 너가 판단해주고 그 이유를 설명해줘
    이전 평가에선 "나빠짐"을 받았어.

    문장 : 저는 데이터 기반으로 문제를 해결하는 데 강점을 가지고 있습니다. 대학 시절, 팀 프로젝트에서 마케팅 데이터를 분석해 제품 타겟층을 재설정했고, 이를 통해 광고 클릭률이 25% 증가한 경험이 있습니다. 이 경험을 통해 제가 지원한 데이터 분석 직무에서 데이터 해석력과 문제 해결 능력을 발휘할 수 있을 것이라 확신합니다. 또한 Python과 Excel을 활용한 실무 분석 경험이 있어 바로 투입될 수 있는 준비가 되어 있습니다.

    요구 사항은 다음과 같아:
    - 판단은 좋아짐, 미미함, 나빠짐 중 하나
    - 이유는 해당 문장부분을 인용해서 설명
    - 아래 JSON 형식으로만 응답해줘 (설명 X)

    형식:
    {{
      "result": {{
        "score":"좋아짐",
        "reason":"해당 답변에서 '광고 클릭률이 25% 증가한 경험'이라는 부분이..."
        }}
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 면접에 대한 답변을 평가하고 피드백해주는 전문적인 AI야"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    except Exception as e:
        return {"error": f"API 요청 실패: {str(e)}"}