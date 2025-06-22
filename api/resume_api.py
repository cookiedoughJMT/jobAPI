from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import re
import os
from fastapi import APIRouter

from prompt.resume_prompt import generate_json_kts_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

resume_api = APIRouter()

class KTSRequest(BaseModel):
    keyword: Optional[str] = None
    personality : Optional[str] = None
# end class

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