from dotenv import load_dotenv
from flask import request_started
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import re
import os
from fastapi import APIRouter

from prompt.resume_prompt import generate_json_q4sg_prompt, generate_json_q3sg_prompt, generate_json_kts_prompt, generate_json_q6sg_prompt, generate_json_q7sg_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

resume_api = APIRouter()

class KTSRequest(BaseModel):
    keyword: Optional[str] = None
    personality : Optional[str] = None
    type : Optional[str] = None
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

# ========================================================================== Q3 sentence Generator API ==========================================================================================

@resume_api.post("/Q3SG")
async def q3sentencegenerator(request: Q3SG):
    prompt = generate_json_q3sg_prompt(request.content, request.major, request.degree)

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

# ========================================================================== Q4 sentence Generator API ==========================================================================================

@resume_api.post("/Q4SG")
async def q4sentencegenerator(request: Q4SG):
    prompt = generate_json_q4sg_prompt(request.content, request.company, request.position, request.workperiod, request.job)

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

# ========================================================================== Q6 sentence Generator API ==========================================================================================

@resume_api.post("/Q6SG")
async def q6sentencegenerator(request: Q6SG):
    prompt = generate_json_q6sg_prompt(request.content, request.achievements)

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

# ========================================================================== KTS API ==========================================================================================

@resume_api.post("/KTS")
async def generate_general_interview(request:KTSRequest):

    prompt = generate_json_kts_prompt(request.keyword, request.personality, request.type)
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

        json_match = re.search(r"(\{.*?\})", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {"sentences": parsed["sentences"]}

        return {"error": "JSON 형식이 감지되지 않았습니다."}


    except Exception as e:
        return {"error": str(e)}
# end KTS API