from dotenv import load_dotenv
from flask import request_started
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import re
import os
from fastapi import APIRouter

from prompt.resume_prompt import generate_json_kts_prompt, generate_json_keywordextractor_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

resume_api = APIRouter()

class KTSRequest(BaseModel):
    keyword: Optional[str] = None
    personality : Optional[str] = None
# end class

class KewordextractorRequest(BaseModel):
    sentence : Optional[str] = None
    personality: Optional[str] = None
    type: Optional[str] = None
# end class

# ========================================================================== keywordExtractor API ==========================================================================================

@resume_api.post("/keywordextractor")
async def keywordextractor(request: KewordextractorRequest):
    prompt = generate_json_keywordextractor_prompt(request.sentence, request.personality, request.type)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "다음 문장에서 핵심 의미를 간파할 수 있는 **중요 단어(키워드)**만 선별하세요."},
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
            return {"keywords": parsed}

        return {"error": "JSON 형식이 감지되지 않았습니다."}

    except Exception as e:
        return {"error": str(e)}


# end KeywordExtractor API


# ========================================================================== KTS API ==========================================================================================

@resume_api.post("/KTS")
async def generate_general_interview(request:KTSRequest):

    prompt = generate_json_kts_prompt(request.keyword, request.personality)
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