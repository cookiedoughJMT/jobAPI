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

def compute_wpm_timeline(words, segment_size=5.0):
    timeline = []
    current_start = 0.0
    end_time = words[-1]["end"] if words else 0

    print(words)

    while current_start < end_time:
        current_end = current_start + segment_size
        count = sum(1 for w in words if current_start <= w["start"] < current_end)
        wpm = int(count / segment_size * 60)
        timeline.append({"time": round(current_start), "wpm": wpm})
        current_start += segment_size

    return timeline

def convert_wpm_timeline_to_speed_pattern(wpm_timeline):
    if not wpm_timeline:
        return []

    end_time = wpm_timeline[-1]["time"] + 5.0  # 마지막 구간 포함

    pattern = []
    for item in wpm_timeline:
        position = int((item["time"] / end_time) * 100)  # 0~100%
        level = item["wpm"]  # 정규화 없이 실제 wpm 값 그대로
        pattern.append({ "position": position, "level": level })

    return pattern

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
        whisper_result = whisper_model.transcribe("uploaded.wav", word_timestamps=True, language='ko')

        # 단어 단위 시간 정보 추출
        words = []
        for seg in whisper_result.get("segments", []):
            words.extend(seg.get("words", []))

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

        print('transcript: ', transcript)
        wpm_timeline = compute_wpm_timeline(words)
        speed_pattern_data = convert_wpm_timeline_to_speed_pattern(wpm_timeline)
        print('speed_pattern_data: ', speed_pattern_data)

        # 3) GPT 프롬프트 -------------------------------------------------------
        prompt = f"""
        다음은 한 사용자의 음성 면접 데이터입니다. 텍스트와 음성 피처를 참고해 아래 JSON 스키마에 **딱 맞춰서** (백틱·주석 없이) 응답하세요.

        ❗️핵심 의무 사항
        1. strengths / improvements / improvementStrategies 항목에 최소 8개씩, 가능하면 10개 이상 작성.
        2. 제목(title)은 서로 다른 관점(논리·전문성·자신감·어조·속도·말투·공감력·시간 관리 등)으로 다양화.
        3. description 에는 구체적 사례를 포함.
        4. improvements 와 improvementStrategies 는 1:1로 대응(같은 순서).

        텍스트가 너무 짧거나 성의가 부족하면 confidence, overallScore 를 10~30 영역으로 낮춰 주세요.
                
        strengths 항목은 최소 8개 이상 해주세요.
        improvements 항목은 최소 8개 이상 해주세요.
        improvementStrategies 항목은 최소 8개 이상 해주세요.                
                
        중복 금지:
        - strengths.title 과 improvements.title 은 절대 중복되지 않도록 작성하세요.
          (중복이 생길 경우, improvements 쪽 항목을 제거)
        
        근거 필수:
        - 모든 description 에서는 실제 음성·텍스트에서 확인 가능한 구체적 근거를 명시하세요.
          근거 없이 추상적 칭찬(예: ‘성실함’, ‘공감성 높음’)은 금지합니다.
          
        문맥만 난해할 경우엔 의미를 보정·추론해 최대한 너그럽게 평가합니다.

        🗣 텍스트
        \"\"\"{transcript}\"\"\"

        🎧 음성 피처 요약
        {feat_txt}
        - 구간별 속도 변화: {json.dumps(wpm_timeline[:5])[:300]}...
        
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

          "metricGrades": {{
            "wordsPerMinute": {{ "grade": "적절/빠름/느림", "comment": "짧은 분석" }},
            "clarity":        {{ "grade": "우수/보통/개선 필요", "comment": "짧은 분석" }},
            "intonation":     {{ "grade": "풍부/단조로움", "comment": "짧은 분석" }},
            "pauseDuration":  {{ "grade": "적절/빠름/과다", "comment": "짧은 분석" }},
            "pronunciation":  {{ "grade": "우수/개선 필요", "comment": "짧은 분석" }},
            "fillers":        {{ "grade": "최소/보통/과다", "comment": "짧은 분석" }}
          }},

          "voicePatterns": {{
            "volumePattern": {{
              "description": "문장",
              "data": [0,0,0,0,0]
            }},
            "speedPattern": {{
              "description": "문장",
              "data": [{{"position":0,"level":0}},{{"position":10,"level":1}},...]
            }},
            "tonePattern": {{
              "description": "문장"
            }}
          }},

          "interviewerComment": "면접관 시점에서 느껴질 전반적 인상 한두 문장",

          "strengths": [
            {{ "title": "짧은 제목(강점)", "description": "구체 사례 포함" }}, ...
          ],
          "improvements": [
            {{ "title": "짧은 제목(개선점)", "description": "구체 사례 포함" }}, ...
          ],
          "improvementStrategies": [
            {{ "title": "짧은 제목(개선 전략)", "description": "개선점과 1:1 대응되는 실행 가능한 전략" }}, ...
          ]
        }}

        지침
        - 모든 JSON 결과는 한국어 또는 숫자로 반환해주세요.
        
        추가 작성 가이드
        - JSON 키·구조 변형 금지, 숫자는 정수 또는 소수 1자리.
        - metricGrades 등급 기준  
          • wordsPerMinute: <80=느림, 80~100/130~160=다소, 100~130=적절, >160=빠름  
          • clarity: ≥80=우수, 60~79=보통, <60=개선 필요  
          • pauseDuration: <0.5s=빠름, 0.5~1.2s=적절, >1.2s=과다  
          • fillers: 0~1=최소, 2~4=보통, ≥5=과다  
        - 음향·언어·논리·태도 등 다각도로 평가.
        - 동일 범주라도 '어투-친근감' vs '어투-전문성'처럼 세부 요소로 분화.
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

        if "voicePatterns" in parsed and "speedPattern" in parsed["voicePatterns"]:
            parsed["voicePatterns"]["speedPattern"]["data"] = speed_pattern_data
        else:
            parsed["voicePatterns"]["speedPattern"] = {
                "description": "시간 흐름에 따른 말하기 속도 변화",
                "data": speed_pattern_data
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
