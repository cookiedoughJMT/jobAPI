from fastapi import UploadFile, File
from fastapi import APIRouter
from dotenv import load_dotenv
from datetime import datetime

import os, json, numpy as np

import whisper
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
from openai import OpenAI

import time

import logging
import re

from pathlib import Path

import ffmpeg

# ─────────────────────── 공통 초기화 ───────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

# ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

# 환경변수 추가
# os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# pydub에 명시적 지정
# AudioSegment.converter = ffmpeg_path

# pydub 내부 경로에도 설정 (추가 방어)
# if not which("ffmpeg"):
#     print("❌ ffmpeg 경로 확인 실패")
# else:
#     print("✅ ffmpeg 경로 확인됨:", which("ffmpeg"))



def convert_webm_to_wav(input_path, output_path):
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), format='wav', acodec='pcm_s16le', ac=1, ar='16000')  # 16kHz mono
        .overwrite_output()
        .run(quiet=True)
    )

# FastAPI 앱
audio_api = APIRouter()

# whisper 모델 (1회만 로드)
whisper_model = whisper.load_model("base")

# 루트 프로젝트 경로(예: main.py 옆) 기준으로 하위 폴더 지정
BASE_DIR       = Path(__file__).resolve().parent
WEBM_DIR       = BASE_DIR / "data" / "webm"
WAV_DIR        = BASE_DIR / "data" / "wav"

# 폴더가 없으면 생성
WEBM_DIR.mkdir(parents=True, exist_ok=True)
WAV_DIR.mkdir(parents=True,  exist_ok=True)

audio_api = APIRouter()
whisper_model = whisper.load_model("base")

def get_timestamp_name() -> str:
    """YYYYMMDD_HHMMSS 형태의 타임스탬프 반환"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ─────────────────────── ① 음성 분석 라우터 ───────────────────────
# audio_api = APIRouter(prefix="/api/audio")

# ────────────────── 헬퍼 함수 ──────────────────

# pitch contour 구하기 librosa(음성분석라이브러리)
def compute_pitch_contour(x, Fs, win, step):
    import librosa

    # librosa.yin은 프레임 단위 pitch 추정 → window/step 사이즈에 맞게 hop 설정
    pitches = librosa.yin(y=x, fmin=50, fmax=300, sr=Fs, frame_length=win, hop_length=step)

    # 0 또는 음수 값 제거 (음성 없는 구간에서 -1 또는 0 나올 수 있음)
    pitch_contour = [float(p) if p > 0 else 0.0 for p in pitches]

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

def compute_wpm_timeline(words, default_segment_size=5.0, short_segment_size=3.0, threshold=30.0):
    timeline = []

    if not words:
        return []

    current_start = 0.0
    end_time = words[-1]["end"]

    # 조건 분기: 10초 미만이면 1초, 30초 미만이면 3초, 아니면 기본값으로 기준 삼아 추출
    if end_time < 10:
        segment_size = 1.0
    elif end_time < threshold:
        segment_size = short_segment_size
    else:
        segment_size = default_segment_size


    print('segment_size: ', segment_size)

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
    start_time = time.perf_counter()  # 시작 시간

    try:
        # 0) 업로드 저장 & 변환 ---------------------------------------------------
        raw = await file.read()

        ts = get_timestamp_name()
        webm_fn = f"{ts}_upload.webm"
        wav_fn = f"{ts}_upload.wav"

        webm_path = WEBM_DIR / webm_fn
        wav_path = WAV_DIR / wav_fn

        # WebM 저장
        with open(webm_path, "wb") as f:
            f.write(raw)

        # WebM → WAV 변환
        # AudioSegment.from_file(webm_path).export(wav_path, format="wav")

        convert_webm_to_wav(webm_path, wav_path)

        # 1) Whisper ---------------------------------------------------
        whisper_result = whisper_model.transcribe(
            str(wav_path), word_timestamps=True, language="ko"
        )
        transcript = whisper_result["text"]

        if len(transcript) > 500:
            transcript = transcript[:500] + "..."

        # 단어 단위 시간 정보 추출
        words = []
        for seg in whisper_result.get("segments", []):
            words.extend(seg.get("words", []))

        # 2) pyAudioAnalysis ---------------------------------------------------
        Fs, x = audioBasicIO.read_audio_file(str(wav_path))
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
        1. strengths / improvements / improvementStrategies 항목에 최소 4개씩, 가능하면 8개 이상 작성.
        2. 제목(title)은 서로 다른 관점(논리·전문성·자신감·어조·속도·말투·공감력·시간 관리 등)으로 다양화.
        3. description 에는 구체적 사례를 포함.
        4. improvements 와 improvementStrategies 는 1:1로 대응(같은 순서).

        텍스트가 너무 짧거나 성의가 부족하면 confidence, overallScore 를 10~30 영역으로 낮춰 주세요.
                
        strengths 항목은 최소 4개 이상 해주세요.
        improvements 항목은 최소 4개 이상 해주세요.
        improvementStrategies 항목은 최소 4개 이상 해주세요.                
                
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
        
        📄 응답은 아래 구조의 JSON 형식으로 작성해주세요:

        - overallScore, clarity, speed, volume, confidence: 0~100 사이의 정수 (confidence는 overallScore와 동일)
        - speechMetrics: 하위 항목 포함
            • wordsPerMinute, clarity, intonation, pauseDuration, pronunciation, fillers
            • 숫자 또는 소수 1자리 (pauseDuration은 초 단위)
        - metricGrades: 각 항목에 대해 아래 형식
            {{ "grade": "등급", "comment": "짧은 설명" }}
            • 등급 범주는 각 항목에 따라 달라짐 (아래 참고)
            • wordsPerMinute, clarity, intonation, pauseDuration, pronunciation, fillers
        - voicePatterns:
            • volumePattern: {{ "description": 문자열, "data": [숫자 배열] }}
            • speedPattern: {{ "description": 문자열, "data": [{{"position": 정수, "level": 정수}}, ...] }}
            • tonePattern: {{ "description": 문자열, "data": [{{"position": 정수, "level": 정수}}, ...] }}
        - interviewerComment: 한두 문장의 면접관 시점 피드백
        - strengths / improvements / improvementStrategies:
            • 각 항목은 4개 이상 작성
            • 형식: {{ "title": "짧은 제목", "description": "..." }}
            • improvements 와 improvementStrategies는 1:1 대응
        
        JSON 키 이름은 정확하게 유지해주세요. 주석이나 불필요한 포맷(백틱 등)은 포함하지 말고, 결과만 출력해주세요.


        지침
        - 모든 JSON 결과는 한국어 또는 숫자로 반환해주세요.
        - 모든 출력 문장은 반드시 ~습니다, ~합니다와 같은 존댓말 종결어미를 사용해주세요.
        - 반말, 명령형, 음슴체 표현(~함, ~됨, ~임, ~하지 않음 등)은 절대 사용하지 마세요.
        
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
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # 4) GPT 응답 파싱 ------------------------------------------------------
        cleaned = clean_md_json(gpt_resp.choices[0].message.content)
        parsed = json.loads(cleaned)          # dict

        # 5) tonePattern.data / speedPattern 직접 계산 후 삽입 -------------------------------
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


        file_size = webm_path.stat().st_size # webm 파일 사이즈 계산

        end_time = time.perf_counter()  # 종료 시간
        duration = round(end_time - start_time, 2)  # 걸린 시간 (초)

        print('걸린시간: ', duration)

        # 6) 최종 반환 ----------------------------------------------------------
        return {
            "transcript": transcript,
            "gpt_feedback": parsed,
            "webmFn": str(webm_fn),
            "webmPath": str(webm_path),
            "wavPath": str(wav_path),
            "webmFileSize": file_size
        }

    except Exception as e:
        logging.exception("analyze_audio error")
        return {"error": str(e)}

# ─────────────────────── 실행 스크립트 ───────────────────────
# uvicorn main:app --reload
