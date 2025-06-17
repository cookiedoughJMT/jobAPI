import pandas as pd
import random
import os

QUESTION_TYPES = ["general", "pressure", "personality", "technical", "situational"]

# ========================================================================== 질문 생성 함수 ==========================================================================================

def generate_qa_block(question_modes):
    qa_tone_map = {
        "general": [
            "따뜻한", "공감적인", "중립적인", "격려하는", "진지한",
            "편안한", "신뢰를 주는", "호기심 어린", "사려 깊은", "친근한"
        ],
        "pressure": [
            "도발적인", "냉철한", "의심스러운", "비꼬는", "차가운",
            "조롱 섞인", "고압적인", "냉소적인", "의도적으로 허를 찌르는"
        ],
        "personality": [
            "공감적인", "따뜻한", "신뢰를 주는", "진지한",
            "사려 깊은", "편안한", "호기심 어린"
        ],
        "technical": [
            "냉철한", "중립적인", "진지한", "호기심 어린", "신뢰를 주는"
        ],
        "situational": [
            "중립적인", "사려 깊은", "호기심 어린", "편안한", "진지한", "친근한"
        ]
    }
    qa_intent_map = {
        "general": [
            "지원자의 성장 경험 파악",
            "직무 관련 경험 공유 유도",
            "지원 동기의 진정성 판단",
            "장단점에 대한 자기 인식 확인",
            "회사 문화 적응력 확인",
            "향후 커리어 방향 확인",
            "자기주도성 및 태도 평가"
        ],
        "pressure": [
            "논리적 사고력 평가",
            "위기 상황 대응력 판단",
            "감정 조절 능력 확인",
            "실수 책임감 검증",
            "비판 수용 여부 판단",
            "이력서 신뢰성 검증",
            "압박 상황에서의 태도 분석"
        ],
        "personality": [
            "소통 방식 및 공감능력 확인",
            "타인 존중 및 협업 태도 평가",
            "실수 시 대응 자세 평가",
            "도덕성 및 윤리적 가치관 점검",
            "자기 성찰 능력 판단",
            "팀 내 갈등 대처 방식 파악",
            "정서적 안정성 및 진정성 평가"
        ],
        "technical": [
            "기술 깊이 확인",
            "실무 응용 능력 검증",
            "문제 해결 접근 방식 평가",
            "개발/업무 도구 숙련도 확인",
            "최근 기술 트렌드에 대한 이해도 평가",
            "기술 선택의 합리성 판단",
            "실제 프로젝트 적용 경험 분석"
        ],
        "situational": [
            "갈등 해결 방식 테스트",
            "리더십 스타일 파악",
            "우선순위 판단 능력 확인",
            "팀워크 상황에서의 태도 평가",
            "도전적 문제 대처 방식 확인",
            "책임감 있는 의사결정 판단",
            "업무 중 돌발 상황 대응 평가"
        ]
    }

    qa_mode_kor = {
        "general": "일반 질문",
        "pressure": "압박 질문",
        "personality": "인성 질문",
        "technical": "기술 질문",
        "situational": "상황형 질문"
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "../data/questions/general_interview_questions.csv")
    general_df = pd.read_csv(csv_path)
    general_pool = general_df["question"].dropna().tolist()

    used_general_questions = []

    qa_block = ""
    for idx, mode in enumerate(question_modes):
        if mode == "general":
            # 중복 없이 랜덤 선택
            available = list(set(general_pool) - set(used_general_questions))
            selected = random.choice(available) if available else "일반 질문 예시"
            used_general_questions.append(selected)
            qa_block += f"{idx + 1}번째 질문: {selected}\n"
        else:
            tone = random.choice(qa_tone_map.get(mode, ["중립적인"]))
            mode_kor = qa_mode_kor.get(mode, mode)
            intent = random.choice(qa_intent_map.get(mode, ["지원자를 파악하는 것"]))
            qa_block += f"{idx+1}번째 질문의 목적은 {intent}이며 {tone} 스타일의 {mode_kor}을 생성해주세요.\n"
    return qa_block.strip()

# ========================================================================== 주요사업 추출 함수 ==========================================================================================

def get_business_area(company_name):
    base_dir = os.path.dirname(__file__)  # 현재 이 파일이 있는 디렉토리
    csv_path = os.path.join(base_dir, '../data/company/jobkorea_companies_industry.csv')
    csv_path = os.path.abspath(csv_path)

    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)

        # company_name과 일치하는 행 찾기
        match = df[df['기업명'] == company_name]

        # 일치하는 행이 있다면 주요사업 컬럼 값을 반환
        if not match.empty:
            business_area = match.iloc[0]['주요사업']
            return business_area
        else:
            print(f"[경고] '{company_name}'에 해당하는 기업명을 찾을 수 없습니다.")
            return None
    except Exception as e:
        print(f"[오류] CSV 파일 처리 중 문제 발생: {e}")
        return None

# ========================================================================== 통합용 질문 5배수 랜덤 생성 함수 ==========================================================================================

def generate_radom_question_types(total_questions):
    assigned = []
    base_count = 0

    while len(assigned) < total_questions:
        # Step 1: 5개 단위 블록 시작
        block_base = QUESTION_TYPES.copy()
        random.shuffle(block_base)  # 순서 무작위화

        # 블록에서 가능한 만큼 추가
        count_to_add = min(5, total_questions - len(assigned))
        assigned.extend(block_base[:count_to_add])
        base_count += count_to_add

        # Step 2: 6~9일 때는 추가로 랜덤하게 유형 삽입
        if base_count % 5 == 0 and len(assigned) < total_questions:
            used_in_extra = set()
            while len(assigned) < base_count + (total_questions % 5):
                remaining_types = list(set(QUESTION_TYPES) - used_in_extra)
                if not remaining_types:
                    break  # 모든 유형을 한 번씩 썼음

                new_type = random.choice(remaining_types)
                assigned.append(new_type)
                used_in_extra.add(new_type)

    return assigned

# ========================================================================== 통합 면접 프롬프트 ==========================================================================================

def generate_json_general_prompt(job, company, q_number):

    roles = [
        "팀 동료",
        "직속 상사",
        "채용 담당자",
        "현업 선배",
        "멘토",
        "인사 담당자"
    ]
    difficulties = ["상", "중", "하"]

    modes = generate_radom_question_types(q_number)
    qa_block = generate_qa_block(modes)

    role = random.choice(roles)
    difficulty = random.choice(difficulties)


    prompt = (
        (f"지원자는 '{company}' 기업에 지원하였으며, 해당 기업에서의 재직 경험은 없습니다.\n" if company else "")
        + (f"지원자가 희망하는 직무는 {job}입니다.\n" if job else "")
        + f"- 면접관 역할: {role}\n"
        + f"- 질문 난이도: {difficulty}\n"
        + "- 각 질문은 실제 인터뷰 상황에서 사용할 수 있도록 자연스럽고 진정성 있게 작성해주세요.\n"
        + "- 각 질문은 서로 완전히 다른 논점과 상황 설정을 가져야 하며, 표현만 바뀐 유사 질문은 제외하세요.\n"
        + "- 이미 문장으로 주어진 질문은 그대로 사용하며, 절대 수정하거나 변형하지 마세요.\n"
        + "- 출력 형식: JSON 배열\n"
        + "```json\n"
        + "{ \"questions\": [\"질문 1\", \"질문 2\", \"질문 3\", \"질문 4\", \"질문 5\",...]}\n"
        + "```"
        + f"{qa_block}\n"
    )

    print(prompt, modes)  # 개발 중 확인용

    return prompt, modes

# ========================================================================== 압박 면접 프롬프트 ==========================================================================================

def generate_json_pressure_prompt(job, company, cnt):
    tones = [
        "도발적인",
        "중립적인",
        "냉철한",
        "친절한 듯 날카로운",
        "의심스러운",
        "비꼬는",
        "차가운",
        "지적하는",
        "불신 가득한",
        "무심한",
        "조롱 섞인",
        "고압적인",
        "냉소적인",
        "의도적으로 허를 찌르는"
    ]
    roles = [
        "면접관",
        "팀장",
        "동료",
        "인사담당자",
        "실무 리더",
        "프로젝트 책임자"
    ]
    intents = [
        "실수 책임감 검증",
        "스트레스 상황 대응력 확인",
        "비판 수용 여부 판단",
        "논리적 사고력 평가",
        "피드백 반응 관찰",
        "지원 동기의 진정성 판단",
        "조직 충성도 평가",
        "갈등 해결 방식 테스트",
        "리더십 스타일 파악",
        "이력서 신뢰성 검증",
        "기술 깊이 확인",
        "실패 대처 방식 평가",
        "위기 대응력 판단",
        "윤리적 가치관 점검",
        "감정 조절 능력 확인"
    ]
    difficulties = ["상", "중", "하"]

    tone = random.choice(tones)
    role = random.choice(roles)
    intent = random.choice(intents)
    difficulty = random.choice(difficulties)

    prompt = (
            f"반드시 {tone} 스타일의 **압박 면접 질문**을 {cnt}개 생성해주세요.\n"
            + (f"지원자는 '{company}' 기업에 지원하였으며, 해당 기업에서의 재직 경험은 없습니다.\n" if company else "")
            + (f"지원자가 희망하는 직무는 {job}입니다.\n" if job else "")
            + f"- 면접관 역할: {role}\n"
            + f"- 질문 목적: {intent}\n"
            + f"- 질문 난이도: {difficulty}\n"
            + f"- 다음 주제를 하나씩 포함한 질문 {cnt}개를 생성하세요:\n"
            + "  1. 실수에 대한 책임 인식\n"
            + "  2. 윤리적 딜레마 상황\n"
            + "  3. 압박 상황에서의 감정 반응\n"
            + "  4. 자기모순적 판단 상황\n"
            + "  5. 조직 내 갈등 상황 대응\n"
            + "- 각 질문은 서로 완전히 다른 논점과 상황 설정을 가져야 하며, 표현만 바뀐 유사 질문은 제외하세요.\n"
            + "- 각 질문은 다음 형식을 따르세요: 상황 설명 → 선택지 제시 → 이유 또는 대응 방식을 묻는 형태.\n"
            + "출력 형식: JSON 배열\n"
            + "```json\n"
            + "{ \"questions\": [\"질문 1\", \"질문 2\", \"질문 3\",...] }\n"
            + "```"
    )

    print(prompt) # 확인용 추후 제거

    return prompt

# ========================================================================== 인성 면접 프롬프트 ==========================================================================================

def generate_json_personality_prompt(job, company):
    # 선택지 정의
    tones = [
        "공감하는", "다정한", "신중한", "유연한", "진솔한", "차분한", "격려하는", "이해심 깊은",
        "조심스러운", "진지한", "따뜻한", "부드러운", "신뢰를 주는", "의도를 파악하려는", "조율하는", "명확하게 확인하는",
        "부담을 주지 않는", "편안하게 이끄는", "적절히 도전적인", "성찰을 유도하는"
    ]

    roles = [
        "면접관", "동료", "팀장", "HR담당자", "멘토", "교육담당자"
    ]

    intents = [
        "지원자의 가치관 파악",
        "인간관계 스타일 확인",
        "스트레스 상황에서의 감정 조절력 평가",
        "실패 이후 태도 점검",
        "타인에 대한 공감 능력 확인",
        "갈등 상황 대처 방식 분석",
        "장기적인 목표와 방향성 파악",
        "조직 문화 적응력 판단",
        "자기인식 및 성장 태도 확인",
        "도덕성 및 윤리 기준 이해",
        "자기 동기부여 방식 파악",
        "비판이나 피드백 수용 태도 확인",
        "다양성에 대한 인식 확인",
        "협업 과정에서의 역할 인식",
        "일과 삶의 균형에 대한 생각 파악",
        "압박 상황에서의 인간적 태도 점검",
        "리더십 경험 및 그에 대한 반성 파악",
        "책임감 있는 태도와 일 처리 기준 확인",
        "의사소통 방식의 유연성 확인"
    ]

    difficulties = ["하", "중"]

    # 무작위 선택
    tone = random.choice(tones)
    role = random.choice(roles)
    intent = random.choice(intents)
    difficulty = random.choice(difficulties)

    # 프롬프트 생성
    prompt = (
            f"{tone} 스타일의 **인성을 파악하기 위한 면접 질문**을 5개 생성해주세요.\n"
            + (f"지원자는 '{company}' 기업에 처음 지원한 상태이며, 재직 경험은 없습니다.\n" if company else "")
            + (f"지원자가 희망하는 직무는 {job}입니다.\n" if job else "")
            + f"- 면접관 역할: {role}\n"
            + f"- 질문 목적: {intent}\n"
            + f"- 질문 난이도: {difficulty}\n"
            + "- 각 질문은 반드시 **서로 다른 주제**를 다뤄야 하며, 표현만 다르고 의미가 유사한 질문은 절대 포함하지 마세요.\n"
            + "- 다음 주제 중 각기 다른 5개를 선택하여 사용하세요: 실패, 갈등, 리더십, 윤리, 책임감, 협업, 판단력, 적응력, 감정조절\n"
            + "- 각 질문은 지원자의 **구체적인 경험과 행동 방식, 가치관**을 반드시 끌어낼 수 있어야 합니다.\n"
            + "- 질문은 단편적인 감정 표현에 머물지 않고, **상황 설명 → 판단 또는 행동 → 결과 또는 교훈**을 유도하는 복합적 구조를 지녀야 합니다.\n"
            + "출력 형식: JSON 배열\n"
            + "```json\n"
            + "{ 'questions': [\"질문 1\", \"질문 2\", \"질문 3\", \"질문 4\", \"질문 5\"],... }\n"
            + "```"
    )

    print(prompt) # 확인용 추후 제거

    return prompt

# ========================================================================== 심츰기술 면접 프롬프트 ==========================================================================================

def generate_json_technical_prompt(job, company):

    business_area = get_business_area(company)

    tones = [
        "분석적인", "논리적인", "객관적인", "비판적인", "검증 중심의", "전문적인"
    ]

    roles = [
        "기술 면접관", "직장 상사", "팀 리더", "CTO", "Tech Lead"
    ]

    intents = [
        "회사 주요 기술 분야에 대한 지원자의 이해도 평가",
        "해당 산업 기술 트렌드에 대한 분석 능력 확인",
        "프로젝트 기반 문제 해결 능력 측정",
        "회사의 사업 모델에 기술적으로 기여할 수 있는지 판단",
        "기술 깊이와 실전 응용 능력 확인",
        "신기술 학습 및 적용 태도 점검"
    ]

    difficulties = ["중", "상", "최상"]

    tone = random.choice(tones)
    role = random.choice(roles)
    intent = random.choice(intents)
    difficulty = random.choice(difficulties)

    prompt = (
        f"{tone} 스타일의 **기술 심층 면접 질문**을 5개 생성해주세요.\n"
        + (f"지원자는 '{company}' 기업에 지원하였으며, 이 회사에서의 재직 경험은 없습니다.\n" if company else "")
        + (f"지원자가 희망하는 직무는 '{job}'입니다.\n" if job else "")
        + (f"'{company}'의 주요 사업 영역은 '{business_area}'입니다.\n" if business_area else "")
        + f"- 면접관 역할: {role}\n"
        + f"- 질문 목적: {intent}\n"
        + f"- 질문 난이도: {difficulty}\n"
        + "- 모든 질문은 해당 회사의 주요 사업 영역과 연결된 주제를 바탕으로 구성해주세요.\n"
        + "- 각 질문은 지원자의 **지식의 깊이, 실무 적용력, 문제 해결력, 학습 태도** 등을 드러낼 수 있도록 하세요.\n"
        + "- 질문은 단편적인 정의 암기가 아니라, **실제 사례 기반 설명, 논리 전개, 비판적 사고**를 유도할 수 있어야 합니다.\n"
        + f"- 생성된 질문은 반드시 '{job}'의 역할(예: 마케팅, 기획 등)에서 실제 담당하거나 분석할 수 있는 수준의 기술 및 전략 주제를 기반으로 해야 합니다.\n"
        + f"- '{job}' 직무의 실무 범위를 벗어나는 R&D, 제조공정, 생산설비 등 기술적 세부 지식에 대한 직접적 질문은 피해주세요.\n"
        + "- 질문 간 주제가 반드시 서로 달라야 하며, 표현만 다른 유사 질문은 포함하지 마세요.\n"
        + "출력 형식: JSON 배열\n"
        + "```json\n"
        + "{ 'questions': [\"질문 1\", \"질문 2\", \"질문 3\", \"질문 4\", \"질문 5\"] }\n"
        + "```"
    )

    print(prompt)  # 확인용 추후 제거
    return prompt

# ========================================================================== 상황 면접 프롬프트 ==========================================================================================

def generate_json_situational_prompt(job, company):
    business_area = get_business_area(company)

    tones = [
        "분석적인", "논리적인", "객관적인", "비판적인", "검증 중심의", "전문적인"
    ]

    roles = [
        "상황면접관", "팀 리더", "직속 상사", "기획 담당자", "PM"
    ]

    intents = [
        "주요 사업 상황에 대한 판단력 평가",
        "갈등 상황에서의 대처 능력 확인",
        "직무 관련 전략적 사고방식 분석",
        "조직 내 협업 및 설득력 평가",
        "새로운 아이디어에 대한 수용 태도 점검"
    ]

    difficulties = ["중", "상", "최상"]

    tone = random.choice(tones)
    role = random.choice(roles)
    intent = random.choice(intents)
    difficulty = random.choice(difficulties)

    prompt = (
        f"{tone} 스타일의 **상황 기반 면접 질문**을 5개 생성해주세요.\n"
        + (f"지원자는 '{company}' 기업에 지원하였으며, 이 회사에서의 재직 경험은 없습니다.\n" if company else "")
        + (f"지원자가 희망하는 직무는 '{job}'입니다.\n" if job else "")
        + (f"'{company}'의 주요 사업 영역은 '{business_area}'입니다.\n" if business_area else "")
        + f"- 면접관 역할: {role}\n"
        + f"- 질문 목적: {intent}\n"
        + f"- 질문 난이도: {difficulty}\n"
        + "- 모든 질문은 회사의 주요 사업과 지원자의 직무에 관련된 현실적인 업무 상황을 가정하여 작성해주세요.\n"
        + "- 각 질문은 지원자가 **상황을 분석하고 판단한 후, 어떤 방식으로 대응할지를 서술할 수 있도록** 구성해야 합니다.\n"
        + "- 질문은 단순한 의견이 아니라, 실제 직무에서 마주할 수 있는 갈등·우선순위 판단·팀 협업 등의 상황을 반영해야 합니다.\n"
        + "- 기술적 세부 지식보다 **의사결정 과정과 사고 방식**에 초점을 맞춰주세요.\n"
        + "- 질문은 반드시 서로 다른 상황 맥락을 다뤄야 하며, 표현만 다르거나 유사한 질문은 포함하지 마세요.\n"
        + "출력 형식: JSON 배열\n"
        + "```json\n"
        + "{ 'questions': [\"질문 1\", \"질문 2\", \"질문 3\", \"질문 4\", \"질문 5\"] }\n"
        + "```"
    )

    print(prompt)

    return prompt

# ========================================================================== 모의 면접 평가 프롬프트 ==========================================================================================

def generate_json_evaluation(
        questions: list[str],
        answers: list[str],
        times: list[float],
        limit_time: float,
        goodorbad_num: int,
        improvment_num: int,
        prev_badpoints: list[str] = None
) -> str:
    q_num = len(questions)

    # 질문/답변 블록 구성
    qa_block = ""
    for i, (question, answer, time) in enumerate(zip(questions, answers, times), 1):
        qa_block += f"질문{i:02}: {question}\n"
        qa_block += f"걸린 시간: {time}분\n"
        qa_block += f"답변{i:02}: {answer}\n\n"

    # prev_badpoints 설명 블록 조건 처리
    prev_badpoint_clause = ""
    if prev_badpoints:
        badpoints_joined = ', '.join(f'"{p}"' for p in prev_badpoints)
        prev_badpoint_clause = f"""
        또한 이전 답변에서 지적받은 {badpoints_joined}에 대해  
        이번 답변에서는 상태가 "좋아짐 / 유사함 / 나빠짐" 중 어떤지, 그 이유와 함께 분석해주세요.
        """
    else:
        prev_badpoint_clause = f"""
        이전 지적사항이 없는 경우, `state01 ~ state0{len(prev_badpoints) if prev_badpoints else 1}` 및 `cause01 ~ cause0{len(prev_badpoints) if prev_badpoints else 1}` 항목에는 null 값을 입력해주세요.
        """

    # 프롬프트 최종 구성
    prompt = f"""
    다음은 {q_num}개의 면접 질문과 그에 대한 지원자의 답변입니다.  
    각 질문별로 개별 평가하는 것이 아니라, **전체 답변의 흐름과 내용 전반을 평가**해주세요.

    평가는 다음 요소들을 기준으로 진행됩니다:
    - 질문 이해도: 0~18점
    - 구체성: 0~18점
    - 자기 성찰: 0~18점
    - 대응 방식: 0~18점
    - 전달력: 0~18점
    - 시간: 0~10점
    - 종합 평균 소요 시간 (3 ~ 10점, {limit_time}분 시간 제한)
    
    점수계산 시 주의할 것
    - 시간 초과 or 0.0분이면 3점
    - 무의미한 답변이 있는 경우 해당 항목에서 0점 또는 감점
    - 반드시 전체 답변 중 2개 이상 무응답또는 무의미한 답변인 경우 총점은 15점을 넘기지 않도록 조정
    
    무의미한 답변 예시
    - "ㅇ","ㅁ","맞습니다.","그렇습니다.","아닙니다."등 한글자 또는 한단어로 쓰는 경우
    - "모르겠습니다"
    - "없습니다" (맥락 없이 단답형으로만 답한 경우)

    총점은 100점 만점으로 계산되며, 그 점수의 이유를 `reason`에 작성해주세요.`reason` 문장은 반드시 정중한 문장으로 마무리해주세요.  
    예: "~이 부족하여 설득력이 떨어집니다.", "~이 개선될 필요가 있습니다.", "~이 필요합니다." 등

    그리고 전반적인 강점과 개선점을 요약 및 설명 형식으로 각각 {goodorbad_num}가지씩 작성해주세요.
    good_summary01 ~ good_summary0{goodorbad_num} 항목과 good_description01 ~ good_description0{goodorbad_num} 항목에는 반드시 **지원자의 긍정적인 측면, 장점, 강점으로 해석할 수 있는 요소만** 작성해주세요.만약 긍정적으로 평가할 요소가 거의 없다면 null로 처리 해주세요.
    
    bad_summary01 ~ bad_summary0{goodorbad_num} 항목과 bad_description01 ~ bad_description0{goodorbad_num} 항목에는 반드시 **지원자의 부족한 부분, 아쉬운 점, 개선이 필요한 요소**를 구체적으로 작성해주세요.만약 특별히 지적할 만한 부족한 점이 없다고 판단되는 경우에는 해당 항목은 `null`로 처리해주세요.
    
    {prev_badpoint_clause}
    마지막으로 각 질문에 대해 하나씩, 총 {q_num}개의 `solution01 ~ solution{q_num}` 항목을 작성해주세요.
    각 solution은 해당 질문에 대한 답변을 보완하거나 개선하기 위한 **구체적이고 실질적인 조언**이어야 합니다.

    또한 마지막에는 `improvment01 ~ improvment0{improvment_num}` 항목으로,  
    **지원자가 다음 면접을 준비할 때 보완해야 할 구체적인 연습 방향이나 태도, 사고 방식 등 실질적인 개선 조언**을 작성해주세요.
      
    소요 시간이 기준인 {limit_time}분과 동일하더라도, 시간 내 **전달력**, **내용의 밀도**, **적절한 속도 조절**이 포함되어야만 시간 항목에서 긍정적 평가를 받을 수 있습니다.  
    단순히 제한 시간을 초과하지 않았다는 이유만으로는 시간 평가에서 높은 점수를 주지 마세요.
    
    또한, 모든 답변 시간이 정확히 제한 시간과 같을 경우, 이는 오히려 **말이 늘어지거나 핵심 없이 시간을 맞춘 것일 수 있음**을 감안하여 주의 깊게 평가하세요.

    반드시 아래 항목이 포함된 **JSON 형식**으로 출력하세요. 설명이나 코드블럭 없이 순수 JSON만 출력해주세요.

    - score
    - reason
    - good_summary01 ~ good_summary0{goodorbad_num}
    - good_description01 ~ good_description0{goodorbad_num}
    - bad_summary01 ~ bad_summary0{goodorbad_num}
    - bad_description01 ~ bad_description0{goodorbad_num}
    - state01 ~ state0{len(prev_badpoints) if prev_badpoints else 1}
    - cause01 ~ cause0{len(prev_badpoints) if prev_badpoints else 1}
    - solution01 ~ solution0{q_num}
    - improvment01 ~ improvment0{improvment_num}

    면접 질문 및 답변은 다음과 같습니다:

    {qa_block}
    """.strip()

    print(prompt)

    return prompt

