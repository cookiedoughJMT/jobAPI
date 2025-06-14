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
# jobAPI

| 폴더명                | 설명                         |
|--------------------|----------------------------|
| 📁 **`api/`**      | 각종 API 세부 코드               |
| 📁 **`data/`**     | 수집된 데이터 파일(csv, json 등) 저장 |
| 📁 **`prompt/`**   | API에 필요한 prompt생성 함수 폴더    |
| 📁 **`config.py`** | API 설정 파일                  |
| 📁 **`main.py`**   | 서버 실행 코드                   |
