from dotenv import load_dotenv
from flask import request_started
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import re
import os
from fastapi import APIRouter

from prompt.resume_prompt import generate_json_q2sg_prompt, generate_json_q4sg_prompt, generate_json_q6sg_prompt, generate_json_q7sg_prompt, generate_json_createresume_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

resume_api = APIRouter()

class Q2SG(BaseModel):
    keyword: Optional[str] = None
    personality : Optional[list[dict]] = None
# end class

class KewordextractorRequest(BaseModel):
    sentence : Optional[str] = None
    personality: Optional[str] = None
    type: Optional[str] = None
# end class

class Q3SG(BaseModel):
    content : Optional[str] = None
    major : Optional[str] = None
    degree : Optional[str] = None
# end class

class Q4SG(BaseModel):
    content : Optional[str] = None
    position : Optional[str] = None
    company : Optional[str] = None
    workperiod : Optional[str] = None
    job : Optional[str] = None
# end class

class Q6SG(BaseModel):
    content : Optional[str] = None
    achievements : list[str] | None
# end class

class Q7SG(BaseModel):
    content : Optional[str] = None
# end class

class Createresume(BaseModel):
    personalities : list[object] = None
    growStroy: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    whenLearn: Optional[str] = None
    company: Optional[str] = None
    workPeriod: Optional[str] = None
    position: Optional[str] = None
    job: Optional[str] = None
    achievements: list[str] = None
    forCompany: Optional[str] = None
# end class

# ========================================================================== Q4 sentence Generator API ==========================================================================================

@resume_api.post("/Q4SG")
async def q3sentencegenerator(request: Q3SG):
    prompt = generate_json_q4sg_prompt(request.content, request.degree, request.major)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 사용자가 입력해준 정보를 토대로 자기소개서를 전문적으로 써주는 AI야"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        # [] 형태의 JSON 배열을 파싱
        json_match = re.search(r"(\[.*?\])", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed

        return {"error": "JSON 형식이 감지되지 않았습니다."}

    except Exception as e:
        return {"error": str(e)}


# end KeywordExtractor API

# ========================================================================== Q7 sentence Generator API ==========================================================================================

@resume_api.post("/Q7SG")
async def q7sentencegenerator(request: Q7SG):
    prompt = generate_json_q7sg_prompt(request.content)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 사용자가 입력해준 정보를 토대로 자기소개서를 전문적으로 써주는 AI야"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        # [] 형태의 JSON 배열을 파싱
        json_match = re.search(r"(\[.*?\])", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed

        return {"error": "JSON 형식이 감지되지 않았습니다."}

    except Exception as e:
        return {"error": str(e)}

# ========================================================================== Final Resume Create API ==========================================================================================

@resume_api.post("/createresume")
async def createresume(request: Createresume):
    prompt = generate_json_createresume_prompt(request.personalities,
                                               request.growStroy,
                                               request.degree, request.major, request.whenLearn, request.company, request.workPeriod, request.position, request.job, request.achievements, request.forCompany)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 사용자가 입력해준 정보를 토대로 자기소개서를 전문적으로 써주는 AI야"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        # ✅ 마크다운 제거
        content = content.replace("```json", "").replace("```", "").strip()

        print(content)

        # ✅ 중괄호 블록만 추출
        json_match = re.search(r"\{.*\}", content, re.DOTALL)

        if json_match:
            parsed = json.loads(json_match.group())
            return parsed

        return {"error": "JSON 객체 형식이 감지되지 않았습니다."}

    except Exception as e:
        return {"error": str(e)}

# ========================================================================== Q2 sentence Generator API  ==========================================================================================

@resume_api.post("/Q2SG")
async def q2sentencegenerator(request:Q2SG):

    prompt = generate_json_q2sg_prompt(request.keyword, request.personality)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "너는 사용자가 주는 키워드와 성격을 통해 사용자의 성장 배경을 소설처럼 길게 써주는 AI야"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        # [] 형태의 JSON 배열을 파싱
        json_match = re.search(r"(\[.*?\])", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed

        return {"error": "JSON 형식이 감지되지 않았습니다."}

    except Exception as e:
        return {"error": str(e)}
# end KTS API