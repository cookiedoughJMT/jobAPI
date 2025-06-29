from fastapi import UploadFile, File
from fastapi import APIRouter
from dotenv import load_dotenv

import os, json, numpy as np

import whisper
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
from openai import OpenAI

from pydub import AudioSegment
from pydub.utils import which

import logging
import re

# ─────────────────────── 공통 초기화 ───────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

# 환경변수 추가
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# pydub에 명시적 지정
AudioSegment.converter = ffmpeg_path

# pydub 내부 경로에도 설정 (추가 방어)
if not which("ffmpeg"):
    print("❌ ffmpeg 경로 확인 실패")
else:
    print("✅ ffmpeg 경로 확인됨:", which("ffmpeg"))

# FastAPI 앱
audio_api = APIRouter()

# whisper 모델 (1회만 로드)
whisper_model = whisper.load_model("base")

# ─────────────────────── ① 음성 분석 라우터 ───────────────────────
# audio_api = APIRouter(prefix="/api/audio")

# ────────────────── 헬퍼 함수 ──────────────────

# pitch contour 구하기 librosa(음성분석라이브러리)
def compute_pitch_contour(signal, fs, win, step):
    import librosa
    pitches, _ = librosa.piptrack(y=signal, sr=fs, n_fft=win, hop_length=step)
    pitch_contour = []
    for i in range(pitches.shape[1]):
        pitch_values = pitches[:, i]
        max_pitch = pitch_values.max()
        pitch_contour.append(max_pitch if max_pitch > 0 else 0)
    return pitch_contour

def extract_tone_pattern_from_pitch(pitch_series):
    result, length = [], len(pitch_series)
    step = max(1, length // 5)
    for i in range(0, length, step):
        segment = pitch_series[i:i + step]
        level = min(100, int(np.std(segment) * 1000))  # 변동성 → 0~100
        result.append({"position": int(i / length * 100), "level": level})
    return result

def clean_md_json(text: str) -> str:
    """ ```json ... ```  감싸기 제거 """
    return re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.DOTALL)


# ────────────────── 메인 엔드포인트 ──────────────────
@audio_api.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 0) 업로드 저장 & 변환 -------------------------------------------------
        raw = await file.read()
        with open("uploaded.webm", "wb") as f:
            f.write(raw)

        AudioSegment.from_file("uploaded.webm").export("uploaded.wav", format="wav")

        # 1) Whisper -----------------------------------------------------------
        transcript = whisper_model.transcribe("uploaded.wav")["text"]

        # 2) pyAudioAnalysis ---------------------------------------------------
        Fs, x = audioBasicIO.read_audio_file("uploaded.wav")
        x = audioBasicIO.stereo_to_mono(x)

        # ⭐️ librosa 사용을 위해 float32로 변환
        x = x.astype(np.float32)

        if np.mean(x ** 2) < 1e-5:
            return {"error": "무음이거나 너무 작음"}

        win = step = int(0.05 * Fs)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, win, step)

        # print("🎧 f_names 확인:", f_names)


        if F.shape[1] == 0:
            return {"error": "유효한 프레임 없음"}

        features = {k: float(np.mean(v)) for k, v in zip(f_names, F)}
        feat_txt = "\n".join(f"- {k}: {v:.4f}" for k, v in features.items())

        # 3) GPT 프롬프트 -------------------------------------------------------
        prompt = f"""
        다음은 한 사용자의 음성 면접 데이터입니다. 텍스트와 음성 피처를 참고해 아래 JSON 스키마에 **딱 맞춰서** (백틱·주석 없이) 응답하세요.
        speedPattern 의 data값은 최대한 여러개를 뽑아주시고 position 의 값은 최소 10씩 차이나게 해주세요.
        strengths, improvements, improvementStrategies 이 항목은 description에 담아야할 내용의 주제를 설명했습니다.
        참고하여 자세히 작성해주세요.
        
        🗣 텍스트
        \"\"\"{transcript}\"\"\"
        
        🎧 음성 피처 요약
        {feat_txt}
        
        {{
          "overallScore":  (10~100 정수),
          "clarity":       (0~100 정수),
          "speed":         (0~100 정수),
          "volume":        (0~100 정수),
          "confidence":    (overallScore 동일),
        
          "speechMetrics": {{
            "wordsPerMinute": 0,
            "clarity":        0,
            "intonation":     0,
            "pauseDuration":  0.0,
            "pronunciation":  0,
            "fillers":        0
          }},
        
          "voicePatterns": {{
            "volumePattern": {{
              "description": "문장",
              "data": [0,0,0,0,0]
            }},
            "speedPattern": {{
              "description": "문장",
              "data": [{{"position":0,"level":0}},...]
            }},
            "tonePattern": {{
              "description": "문장"
            }}
          }},
        
         "strengths": [
            {{ "title": "짧은 제목(강점)", "description": "전체 강점을 설명해주세요" }},
            {{ "title": "짧은 제목(강점)", "description": "디테일한 강점을 설명해주세요" }}, ...
          ],
          "improvements": [
            {{ "title": "짧은 제목(개선점)", "description": "전체 개선점을 설명해주세요" }},
            {{ "title": "짧은 제목(개선점)", "description": " 디테일한 개선점을 설명해주세요" }}, ...
          ],
          "improvementStrategies": [
            {{ "title": "짧은 제목(개선 전략)", "description": "한 문장 설명" }}, ...
          ]
        }}
        
        지침
        - 위 JSON 키·구조 변형 금지, 숫자는 정수 또는 소수 1자리.
        """

        gpt_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # 4) GPT 응답 파싱 ------------------------------------------------------
        cleaned = clean_md_json(gpt_resp.choices[0].message.content)
        parsed = json.loads(cleaned)          # dict

        # 5) tonePattern.data 직접 계산 후 삽입 -------------------------------
        pitch_series = compute_pitch_contour(x, Fs, win, step)
        tone_pattern = extract_tone_pattern_from_pitch(pitch_series)

        if "voicePatterns" in parsed and "tonePattern" in parsed["voicePatterns"]:
            parsed["voicePatterns"]["tonePattern"]["data"] = tone_pattern
        else:
            parsed["voicePatterns"]["tonePattern"] = {
                "description": "응답 내 tonePattern 누락됨, 직접 추가됨",
                "data": tone_pattern
            }

        # 6) 최종 반환 ----------------------------------------------------------
        return {
            "transcript": transcript,
            "features": features,
            "gpt_feedback": parsed     # ← 이미 dict!
        }

    except Exception as e:
        logging.exception("analyze_audio error")
        return {"error": str(e)}

# ─────────────────────── 실행 스크립트 ───────────────────────
# uvicorn main:app --reload
