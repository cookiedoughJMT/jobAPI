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

#=========================================이전 평가 분석 block생성 함수 ===============================

def generate_prev_badpoint_clause(prev_badpoints: list[str]) -> str:
    if prev_badpoints:
        clause = ""
        for i, point in enumerate(prev_badpoints, start=1):
            clause += f'-state{i:02}: 해당 지원자는 이전 답변에서 "{point}" 항목에 대해 지적받으셨습니다. 이번 답변을 참고하여 얼마나 개선되었는지를 (좋아짐 / 유사함 / 나빠짐) 중 하나로 선택해 주세요.\n'
            clause += f'-cause{i:02}: 위 상태(state{i:02}) 판단의 근거를 지원자의 이번 답변에서 구체적인 문장과 함께 설명해 주세요.\n'
        #end for
        clause += f"위 state와 cause는 반드시 각각 {len(prev_badpoints)}개씩 작성해 주세요. 누락 없이 출력되어야 하며, 동일하거나 반복적인 내용 없이 다양하게 작성해 주세요"
        return clause
    else:
        return "-state: 반드시null반환\n-cause: 반드시null반환\n"


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

    # 동적으로 예시 생성
    example_questions = ', '.join([f'"질문 {i+1}"' for i in range(q_number)])

    prompt = (
        (f"지원자는 '{company}' 기업에 지원하였으며, 해당 기업에서의 재직 경험은 없습니다.\n" if company else "")
        + (f"지원자가 희망하는 직무는 {job}입니다.\n" if job else "")
        + f"- 면접관 역할: {role}\n"
        + f"- 질문 난이도: {difficulty}\n"
        + f"- 각 질문은 실제 인터뷰 상황에서 사용할 수 있도록 자연스럽고 진정성 있게 작성해주세요.\n"
        + f"- 각 질문은 서로 완전히 다른 논점과 상황 설정을 가져야 하며, 표현만 바뀐 유사 질문은 제외하세요.\n"
        + f"- 이미 문장으로 주어진 질문은 그대로 사용하며, 절대 수정하거나 변형하지 마세요.\n"
        + f"- 반드시 정확히 {q_number}개의 질문만 생성하세요. 예시 개수도 {q_number}개로 맞추세요. 그 이상/이하로 생성하지 마세요.\n"
        + "- 출력 형식: JSON 배열\n"
        + "```json\n"
        + f"{{ \"questions\": [{example_questions}] }}\n"
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
            + "너무 길지 않게 생성 하되 아래 요구 사항을 최대한 수용 하여 질문을 만들어 주시고 반드시 undefinded를 반환 하지 말아 주세요."
            + (f"지원자는 '{company}' 기업에 지원하였으며, 해당 기업에서의 재직 경험은 없습니다.\n" if company else "")
            + (f"지원자가 희망하는 직무는 {job}입니다.\n" if job else "")
            + f"- 면접관 역할: {role}\n"
            + f"- 질문 목적: {intent}\n"
            + f"- 질문 난이도: {difficulty}\n"
            + f"- 반드시 각 질문마다 다음 주제(1번 부터 5번까지)중에 하나를 골라 이를 포함한 질문 {cnt}개를 생성하세요:\n"
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

def generate_json_personality_prompt(job, company, cnt):
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
            f"{tone} 스타일의 **인성을 파악하기 위한 면접 질문**을 {cnt}개 생성해주세요.\n"
            + (f"지원자는 '{company}' 기업에 처음 지원한 상태이며, 재직 경험은 없습니다.\n" if company else "")
            + (f"지원자가 희망하는 직무는 {job}입니다.\n" if job else "")
            + f"- 면접관 역할: {role}\n"
            + f"- 질문 목적: {intent}\n"
            + f"- 질문 난이도: {difficulty}\n"
            + "- 각 질문은 반드시 **서로 다른 주제**를 다뤄야 하며, 표현만 다르고 의미가 유사한 질문은 절대 포함하지 마세요.\n"
            + "- 다음 주제 중 각기 다른 {cnt}개를 선택하여 사용하세요: 실패, 갈등, 리더십, 윤리, 책임감, 협업, 판단력, 적응력, 감정조절\n"
            + "- 각 질문은 지원자의 **구체적인 경험과 행동 방식, 가치관**을 반드시 끌어낼 수 있어야 합니다.\n"
            + "- 질문은 단편적인 감정 표현에 머물지 않고, **상황 설명 → 판단 또는 행동 → 결과 또는 교훈**을 유도하는 복합적 구조를 지녀야 합니다.\n"
            + "출력 형식: JSON 배열\n"
            + "```json\n"
            + f"{{ 'questions': [\"질문 1\", ... , \"질문 {cnt}\"] }}\n"
            + "```"
    )

    print(prompt) # 확인용 추후 제거

    return prompt

# ========================================================================== 심츰기술 면접 프롬프트 ==========================================================================================

def generate_json_technical_prompt(job, company, cnt):
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
        f"{tone} 스타일의 **기술 심층 면접 질문**을 {cnt}개 생성해주세요.\n"
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
        + f"{{ 'questions': [\"질문 1\", ... , \"질문 {cnt}\"] }}\n"
        + "```"
    )

    print(prompt)  # 확인용 추후 제거
    return prompt

# ========================================================================== 상황 면접 프롬프트 ==========================================================================================

def generate_json_situational_prompt(job, company, cnt):
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
        f"{tone} 스타일의 **상황 기반 면접 질문**을 {cnt}개 생성해주세요.\n"
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
        + f"{{ 'questions': [\"질문 1\", ... , \"질문 {cnt}\"] }}\n"
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
        alternativemode: str,
        modes: list[str],
        prev_badpoints: list[str] = None,
) -> str:
    q_num = len(questions)

    # 질문/답변 블록 구성
    qa_block = {"일반":f"""
    아래의 점수기준을 참고하여 아래 압박 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
    - 질문 적합성 (30점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
    - 구체성 (25점): 실제 경험이나 구체적 사례가 포함되었는가?
    - 전달력 (15점): 논리적이고 명확한 문장으로 표현되었는가?
    - 자기성찰/인식 (15점): 자신의 태도, 행동에 대한 통찰이 있는가?
    - 시간 적절성 (15점): 답변 시간(분)이 충분했는가? (제한 시간은 {limit_time}분이다.)
    """,
                "압박":f"""
    아래의 점수기준을 참고하여 아래 압박 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
    - 질문 적합성 (25점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
    - 구체성 (20점): 실제 경험이나 구체적 사례가 포함되었는가?
    - 전달력 (20점): 논리적이고 명확한 문장으로 표현되었는가?
    - 자기성찰/인식 (10점): 자신의 태도, 행동에 대한 통찰이 있는가?
    - 시간 적절성 (25점): 답변 시간(분)이 충분했는가? (제한 시간은 {limit_time}분이다.)
    """,

                "인성":f"""
     아래의 점수기준을 참고하여 아래 인성 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
    - 질문 적합성 (30점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
    - 구체성 (20점): 실제 경험이나 구체적 사례가 포함되었는가?
    - 전달력 (20점): 논리적이고 명확한 문장으로 표현되었는가?
    - 자기성찰/인식 (20점): 자신의 태도, 행동에 대한 통찰이 있는가?
    - 시간 적절성 (10점): 답변 시간(분)이 충분했는가? (제한 시간은 {limit_time}분이다.)
    """,

                "기술":f"""
      아래의 점수기준을 참고하여 아래 기술 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
    - 질문 적합성 (30점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
    - 구체성 (30점): 실제 경험이나 구체적 사례가 포함되었는가?
    - 전달력 (15점): 논리적이고 명확한 문장으로 표현되었는가?
    - 자기성찰/인식 (10점): 자신의 태도, 행동에 대한 통찰이 있는가?
    - 시간 적절성 (15점): 답변 시간(분)이 충분했는가? (제한 시간은 {limit_time}분이다.)
    """,

                "상황":f"""
      아래의 점수기준을 참고하여 아래 상황 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
    - 질문 적합성 (25점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
    - 구체성 (25점): 실제 경험이나 구체적 사례가 포함되었는가?
    - 전달력 (15점): 논리적이고 명확한 문장으로 표현되었는가?
    - 자기성찰/인식 (20점): 자신의 태도, 행동에 대한 통찰이 있는가?
    - 시간 적절성 (15점): 답변 시간(분)이 충분했는가? (제한 시간은 {limit_time}분이다.)
    """}

    qa_block_nonetime = {"일반": f"""
        아래의 점수기준을 참고하여 아래 압박 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
        - 질문 적합성 (35점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
        - 구체성 (30점): 실제 경험이나 구체적 사례가 포함되었는가?
        - 전달력 (20점): 논리적이고 명확한 문장으로 표현되었는가?
        - 자기성찰/인식 (15점): 자신의 태도, 행동에 대한 통찰이 있는가?
        """,
                "압박": f"""
        아래의 점수기준을 참고하여 아래 압박 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
        - 질문 적합성 (35점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
        - 구체성 (25점): 실제 경험이나 구체적 사례가 포함되었는가?
        - 전달력 (25점): 논리적이고 명확한 문장으로 표현되었는가?
        - 자기성찰/인식 (15점): 자신의 태도, 행동에 대한 통찰이 있는가?
        """,

                "인성": f"""
         아래의 점수기준을 참고하여 아래 인성 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
        - 질문 적합성 (35점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
        - 구체성 (20점): 실제 경험이나 구체적 사례가 포함되었는가?
        - 전달력 (20점): 논리적이고 명확한 문장으로 표현되었는가?
        - 자기성찰/인식 (25점): 자신의 태도, 행동에 대한 통찰이 있는가?
        """,

                "기술": f"""
          아래의 점수기준을 참고하여 아래 기술 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
        - 질문 적합성 (35점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
        - 구체성 (35점): 실제 경험이나 구체적 사례가 포함되었는가?
        - 전달력 (20점): 논리적이고 명확한 문장으로 표현되었는가?
        - 자기성찰/인식 (10점): 자신의 태도, 행동에 대한 통찰이 있는가?
        """,

                "상황": f"""
          아래의 점수기준을 참고하여 아래 상황 면접 질문에 대한 답변을 1~100점 사이에 정수형태로 평가해줘
        - 질문 적합성 (30점): 질문의 의도를 제대로 파악하고 핵심을 답변했는가?
        - 구체성 (30점): 실제 경험이나 구체적 사례가 포함되었는가?
        - 전달력 (20점): 논리적이고 명확한 문장으로 표현되었는가?
        - 자기성찰/인식 (20점): 자신의 태도, 행동에 대한 통찰이 있는가?
        """}

    qa_customblock = ""

    selected_qa_block = qa_block_nonetime if limit_time == 0.0 else qa_block

    if alternativemode == "커스텀":
        for i in range(q_num):
            mode = modes[i]
            block = selected_qa_block[mode]
            block += f"""
                    질문: {questions[i]}
                    """
            block += f"""걸린 시간: {times[i]}분""" if limit_time != 0.0 else ""
            block += f"""
                    답변: {answers[i]}
                    """
            qa_customblock += block + "\n"

    else:
        for i in range(q_num):
            block = selected_qa_block[alternativemode]
            block += f"""
                    질문: {questions[i]}
                    """
            block += f"""걸린 시간: {times[i]}분""" if limit_time != 0.0 else ""
            block += f"""
                    답변: {answers[i]}
                    """
            qa_customblock += block + "\n"
    # end if

    good_bad_block = ""
    for i in range(goodorbad_num):
        idx = f"{i + 1:02}"
        good_bad_block += f"-good_summary{idx}: 전체적으로 봤을 때 지원자의 답변의 강점을 한 줄로 요약해 주세요. **답변에 강점이 전혀 없는 경우 'null'로 작성해 주세요.**\n"
        good_bad_block += f"-good_description{idx}: 위 good_summary{idx}에 대한 구체적인 이유를 친절하고 정중하게 작성해 주세요. **good_summary가 'null'인 경우 'null'로 작성해 주세요.**\n"
        good_bad_block += f"-bad_summary{idx}: 전체적으로 봤을 때 지원자의 답변에서 개선이 필요한 점을 한 줄로 요약해 주세요. **개선할 점이 전혀 없는 경우 'null'로 작성해 주세요.**\n"
        good_bad_block += f"-bad_description{idx}: 위 bad_summary{idx}에 대한 구체적인 이유를 친절하고 정중하게 작성해 주세요. **bad_summary가 'null'인 경우 'null'로 작성해 주세요.**\n"

    good_bad_block += (
        f"위 항목들은 반드시 각각 총 {goodorbad_num}개씩 작성해 주세요. 누락 없이 출력되어야 하며, "
        f"동일하거나 반복적인 내용 없이 다양하게 작성해 주세요.\n"
    )

    improvment_block = ""
    for i in range(improvment_num):
        idx = f"{i + 1:02}"
        improvment_block += f"-improvment{idx}: 지원자가 다음 면접에서 100점을 받기 위해 어떤 연습을 하면 좋을지 구체적이고 친절하게 작성해 주세요.\n"
    improvment_block += f"위 improvment 항목은 반드시 총 {improvment_num}개로 구성되어야 하며, 누락 없이 출력되어야 하며, 동일하거나 반복적인 내용 없이 다양하게 작성해 주세요\n"
    # end for

    solution_block = ""
    for i in range(q_num):
        idx = f"{i + 1:02}"
        solution_block += f"-solution{idx}: 지원자에게 자연스럽고 실질적인 피드백을 제공해 주세요. 반복된 문구 없이, 직접 조언하듯 작성해 주세요.\n"
    solution_block += f"위 solution 항목은 반드시 총 {q_num}개로 구성되어야 하며, 누락 없이 출력되어야 하며, 동일하거나 반복적인 내용 없이 다양하게 작성해 주세요\n"
    # end for


    score_lines = ""
    for i in range(1, q_num + 1):
        score_lines += f'-score{i:02}: 해당 질문({i})에 대한 점수를 1~100 사이의 정수로 평가해 주세요.\n'
    score_lines += f'-score: 위 score01 ~ score{q_num:02} 항목의 평균값을 계산하여 최종 점수로 작성해 주세요.\n'
    # end for

    # 프롬프트 최종 구성
    prompt = f"""
    {qa_customblock}
    
    아래는 실제로 응답해야 하는 항목에 관한 설명이며 반드시 모두 **지원자에게 직접 전달하듯** 작성하고 어투는 **친절하고 정중한 존댓말**을 사용하고, **무조건적인 칭찬이나 비난보다는 개선 방향과 구체적인 예시를 중심으로** 서술해줘
    - 예: "~할 수 있었을 것입니다.", "~라는 점에서 강점이 있습니다.", "다음에는 ~하는 연습을 해보시면 좋겠습니다."
    - 지원자의 답변이 'ㅇ', '...', '잘 모르겠습니다', '무의미한 단어 반복' 또는 이에 준하는 내용일 경우, 해당 질문은 10점 이하로 평가해 주세요.
    - 반드시 답변의 **내용과 논리적 연결성**을 판단 기준으로 삼고, 길이나 문장 수가 아니라 의미를 기반으로 점수를 부여해 주세요.

    
    {score_lines}
    -reason: 각 질문마다 평가했던 점수가 어떤 이유로 나오게 된 것인지 자세하게 알려줘
    {good_bad_block}
    {generate_prev_badpoint_clause(prev_badpoints)}
    {solution_block}
    {improvment_block}
    
    아래 JSON 형식의 응답을 반환하되, 반드시 **설명이나 코드블럭 없이 순수 JSON 문자열만 출력**하고 모든 응답은 반드시 자연스러운 존댓말(높임말) 형태로 작성해
    ```json
      {{
        "score": 87,
        "reason": "질문의 의도를.....",
        "good_summary01": "강점 요약01",
        ...
        "good_description01": "강점 설명01",
        ...
        "bad_summary01": "개선점 요약01",
        ...
        "bad_description01": "개선점 설명01",
        ...
        "state01":"유사함",
        ....
        "cause01":"이전 답변에 대한 평가에 대한 이유01",
        ....
        "solution01":"질문01에 대한 방향성01",
        "solution02":"질문02에 대한 방향성02",
        ...
        "improvment01":"100점을 향한 연습 추천01",
        "improvment02":"100점을 향한 연습 추천02",
        ....
      }}
    """.strip()

    print(prompt)

    return prompt

