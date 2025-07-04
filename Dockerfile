FROM python:3.10

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# requirements.txt만 먼저 복사 -> 캐시
COPY requirements.txt .

# pip install -> 이 레이어는 requirements.txt 변경 없으면 재사용
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 코드 복사
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
