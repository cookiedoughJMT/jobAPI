압박 면접 질문 생성

POST api/interview/pressure_interview

직무와 회사를 기반으로 압박 면접 질문을 생성합니다.

✅ 요청 형식

{
  "job_role": "데이터 분석가",
  "company": "카카오"
}

✅ 응답 예시

{
    "questions": [
        "질문01","질문02","질문03".....
    ]
}


인성 면접 질문 생성

POST api/interview/personality_interview

✅ 요청 형식

{
  "job_role": "프론트엔드 개발자",
  "company": "네이버"
}

✅ 응답 예시

{
    "question":["질문01","질문02","질문03"...] 
}

모의 면접 답변 평가

POST api/interview/evaluation

✅요청 형식

{
  "questions": [
    "협업 중 갈등을 해결한 경험이 있나요?",
    "압박 상황에서도 집중했던 경험이 있나요?"
  ],
  "answers": [
    "디자인 논쟁 중 중재하며 회의 조율로 해결했습니다.",
    "배포 중 오류 상황에서 침착하게 원인 분석하고 해결했습니다."
  ],
  "times": [4.5, 5.0],
  "limit_time": 5.0,
  "goodorbad_num": 3,
  "improvment_num": 3,
  "prev_badpoints": ["구체성 부족", "핵심 전달 부족", "감정 조절 부족"]
}

✅ 응답 예시

{
  "score": 82,
  "reason": "답변의 전반적인 구체성은 높았으나 자기 성찰이 부족하였습니다.",
  "good_summary01": "...",
  "bad_summary01": "...",
  "state01": "좋아짐",
  "solution01": "질문1에 대해 더 구체적 예시를 들어 설명하세요.",
  "improvment01": "자신의 경험에서 감정 조절 과정을 더 풀어 설명하는 훈련이 필요합니다."
}