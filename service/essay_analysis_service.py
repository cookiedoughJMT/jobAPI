import json
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import numpy as np

from config import (
    OPENAI_API_KEY, 
    SIMILARITY_THRESHOLD, 
    SENTENCE_MODEL, 
    GPT_MODEL, 
    GPT_TEMPERATURE
)

logger = logging.getLogger(__name__)

class EssayAnalysisService:
    def __init__(self):
        """자소서 분석 서비스 초기화"""
        self.sentence_model = SentenceTransformer(SENTENCE_MODEL)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
    def group_similar_sentences(self, sentences: List[str]) -> List[List[int]]:
        """
        의미 유사도 기반으로 문장들을 그룹핑
        
        Args:
            sentences: 분석할 문장 리스트
            
        Returns:
            List[List[int]]: 각 그룹의 문장 인덱스들
        """
        if not sentences:
            return []
            
        logger.info(f"=== 문장 그룹핑 시작 ===")
        logger.info(f"전체 문장 수: {len(sentences)}")
        logger.info(f"유사도 임계값: {SIMILARITY_THRESHOLD}")
        
        # 모든 문장 로그 출력
        for i, sentence in enumerate(sentences):
            logger.info(f"문장 [{i}]: {sentence}")
            
        # 문장 임베딩 생성
        logger.info("문장 임베딩 생성 중...")
        embeddings = self.sentence_model.encode(sentences, convert_to_tensor=True)
        
        # 코사인 유사도 계산
        logger.info("코사인 유사도 계산 중...")
        cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # 유사도 매트릭스 로그 출력 (임계값 이상인 것만)
        logger.info("=== 유사도 매트릭스 (임계값 이상만) ===")
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = cosine_scores[i][j]
                if similarity > SIMILARITY_THRESHOLD:
                    logger.info(f"문장 [{i}] ↔ 문장 [{j}]: 유사도 {similarity:.4f} ⭐ (임계값 초과)")
                    logger.info(f"  [{i}]: {sentences[i][:50]}...")
                    logger.info(f"  [{j}]: {sentences[j][:50]}...")
        
        # 유사도 기반 그룹핑
        logger.info("=== 그룹핑 시작 ===")
        visited = set()
        groups = []
        
        for i in range(len(sentences)):
            if i in visited:
                continue
                
            group = [i]
            visited.add(i)
            similar_found = []
            
            for j in range(i + 1, len(sentences)):
                if j not in visited and cosine_scores[i][j] > SIMILARITY_THRESHOLD:
                    group.append(j)
                    visited.add(j)
                    similar_found.append((j, cosine_scores[i][j]))
                    
            groups.append(group)
            
            # 그룹 형성 로그
            if len(group) > 1:
                logger.info(f"📌 그룹 {len(groups)} 형성: 문장 {group} (총 {len(group)}개)")
                logger.info(f"  대표 문장 [{i}]: {sentences[i]}")
                for j, sim in similar_found:
                    logger.info(f"  유사 문장 [{j}]: {sentences[j]} (유사도: {sim:.4f})")
            else:
                logger.info(f"📌 단독 문장: [{i}] {sentences[i][:50]}...")
                
        logger.info(f"=== 그룹핑 완료 ===")
        logger.info(f"총 그룹 수: {len(groups)}")
        logger.info(f"유사 그룹 수: {len([g for g in groups if len(g) > 1])}")
        logger.info(f"단독 문장 수: {len([g for g in groups if len(g) == 1])}")
        
        # 최종 그룹 요약
        similar_groups = [g for g in groups if len(g) > 1]
        single_groups = [g for g in groups if len(g) == 1]
        
        logger.info("=== 최종 그룹 요약 ===")
        for i, group in enumerate(similar_groups, 1):
            logger.info(f"유사 그룹 {i}: {len(group)}개 문장 -> 1개로 축약")
            for idx in group:
                logger.info(f"  문장 [{idx}]: {sentences[idx]}")
                
        logger.info(f"단독 문장: {len(single_groups)}개")
        total_original = len(sentences)
        total_after_grouping = len(similar_groups) + len(single_groups)
        removed_count = total_original - total_after_grouping
        logger.info(f"문장 수 변화: {total_original}개 -> {total_after_grouping}개 (제거: {removed_count}개)")
            
        return groups
    
    def prepare_grouped_sentences_for_gpt(self, sentences: List[str], groups: List[List[int]]) -> List[Dict[str, Any]]:
        """
        GPT에게 전달할 그룹화된 문장 데이터 준비
        
        Args:
            sentences: 원본 문장 리스트
            groups: 그룹핑된 문장 인덱스들
            
        Returns:
            List[Dict]: 각 그룹의 문장들과 메타데이터
        """
        grouped_data = []
        
        for group_index, group in enumerate(groups):
            if group:  # 그룹이 비어있지 않은 경우
                group_sentences = [sentences[i] for i in group]
                grouped_data.append({
                    "group_id": group_index + 1,
                    "sentences": group_sentences,
                    "sentence_indices": group,
                    "is_similar_group": len(group) > 1  # 유사한 문장이 2개 이상인 그룹
                })
        
        return grouped_data
    
    def build_gpt_prompt(self, grouped_data: List[Dict[str, Any]]) -> str:
        """
        GPT 분석을 위한 프롬프트 구성 (그룹화된 문장 기반)
        
        Args:
            grouped_data: 그룹화된 문장 데이터
            
        Returns:
            str: 구성된 프롬프트
        """
        # 그룹별 문장 정보 구성
        similar_groups_text = ""
        single_sentences_text = ""
        
        for group in grouped_data:
            group_id = group["group_id"]
            sentences = group["sentences"]
            is_similar = group["is_similar_group"]
            
            if is_similar:
                similar_groups_text += f"\n📌 그룹 {group_id} (의미 유사한 문장들 - 최적 1개만 선택):\n"
                for i, sentence in enumerate(sentences, 1):
                    similar_groups_text += f"  {group_id}-{i}. {sentence}\n"
            else:
                single_sentences_text += f"\n📌 단독 문장 {group_id}:\n"
                single_sentences_text += f"  - {sentences[0]}\n"
        
        prompt = f"""당신은 전문적인 자기소개서 컨설턴트입니다. 
다음 문장들을 분석하고 개선해 주세요.

🚨 **핵심 원칙**: corrections는 오직 아래 2가지 경우에만 생성하세요:
1. **중복 제거**: 의미 유사 그룹에서 선택되지 않은 문장들 (improved = "", reason = "'[선택된 문장 앞부분]...'와 의미상 중복 내용이라 삭제되었습니다")
2. **실제 개선 필요**: 맞춤법, 문법, 표현에 명확한 오류가 있는 문장들만

**절대 하지 마세요**: 이미 완벽한 문장을 corrections에 포함하는 것

📌 분석 및 개선 기준:
1. **의미 유사 그룹**: 각 그룹에서 가장 좋은 문장 1개만 선택하고, 나머지는 중복 제거

2. **문법 맞춤법 개선** (선택된 문장 + 단독 문장):
   ✅ **맞춤법 및 띄어쓰기**
   - 철자 오류 수정
   - 올바른 띄어쓰기 적용  
   - 한글 맞춤법 규칙 준수
   
   ✅ **문법 및 어법**
   - 주어와 서술어의 호응 확인
   - 시제의 일관성 유지
   - 올바른 어순 적용
   - 조사 사용법 교정
   
   ✅ **문체 및 표현**
   - 자연스러운 문어체로 통일
   - 적절한 경어법 사용
   - 축약어나 비속어 제거
   - 문장 길이 조절로 가독성 향상

3. **문맥 고려 연결성**: 앞뒤 문장과의 자연스러운 흐름 확보
   - 적절한 접속어 사용
   - 논리적 순서와 구조 고려
   - 전체 자소서의 일관성 유지

4. **부족한 영역 분석**: 다음 항목들을 체계적으로 검토하여 **자소서에 아직 언급되지 않은** 부족한 영역만 도출
   
   ⚠️ **중요**: 
   - 이미 자소서에 충분히 언급된 영역은 부족한 영역으로 분류하지 않음
   - **억지로 부족한 영역을 만들지 마세요**: 모든 영역이 충분히 다뤄졌다면 missing_areas는 빈 배열 []로 반환
   - **완성도 높은 자소서**: 4가지 영역이 모두 잘 갖춰진 경우 부족한 영역 없음으로 판단
   
   ✅ **직무 역량 표현**
   - 문제 상황을 인식하고 해결한 사례의 명확한 서술
   - 팀워크, 협업, 커뮤니케이션 능력이 드러나는 경험  
   - 해당 직무와 연관된 기술력, 전공, 자격증, 실무 경험의 자연스러운 표현
   
   ✅ **성실성과 태도**
   - 꾸준한 노력, 장기간 지속된 활동, 자기계발 노력 표현
   - 외부 교육, 학습, 자격증 취득 등 성실한 준비 태도
   - 맡은 일을 책임감 있게 끝까지 수행한 사례
   
   ✅ **리더십과 도전정신**
   - 조직을 이끌거나 조율한 리더 경험
   - 실패 극복이나 새로운 시도를 한 도전 경험
   - 갈등 해결 또는 구성원 간 문제 조정 사례
   
   ✅ **결과 및 성과 중심 표현**
   - 경험의 결과를 수치나 지표(퍼센트, 증가율, 수량 등)로 표현
   - 타인의 피드백을 수용하고 개선한 사례
   - 활동이나 프로젝트의 결과에 대한 회고 또는 반성

📌 작업 순서:
1. 의미 유사 그룹에서 최적 문장 선택 (나머지는 중복 제거로 corrections에 추가)
2. 선택된 문장들과 단독 문장들을 **원본 순서대로 배열하여 전체 흐름 파악**
3. **자소서 전체 내용 분석**: 언급된 경험, 기술, 전공, 관심사, 활동 등을 파악
4. 각 문장을 체계적으로 개선 (개선사항이 있으면 반드시 corrections에 추가)
   - 맞춤법, 띄어쓰기, 문법 오류 수정
   - 문체를 자연스러운 문어체로 통일
   - 앞뒤 문맥을 고려한 연결성 확보
5. **신중한 부족 영역 판단**: 전체 자소서를 위 4가지 핵심 영역으로 분석
   - **각 영역이 충분히 다뤄진 경우**: missing_areas = [] (빈 배열)
   - **정말 부족한 영역만**: 개인 맞춤형 개선 제안 제공 (억지로 만들지 않기)

📌 개선 시 주의사항:
⚠️ **절대 변경하지 말 것**
- 원본의 의미와 핵심 내용
- 개인적인 경험이나 구체적 사례
- 전체적인 문단 구조와 순서
- 지원자의 개성과 특색

✅ **개선 대상**
- 언어적 오류 (맞춤법, 문법, 띄어쓰기)
- 어색한 표현과 구어체
- 문장 간 연결성과 흐름

📌 corrections 생성 규칙 (중요!):
- **중복 제거만**: original = "제거될 문장", improved = "", reason = "'[선택된 문장의 앞부분 15-20글자]...'와 의미상 중복 내용이라 삭제되었습니다"
- **실제 개선이 필요한 경우만**: original = "원본 문장", improved = "개선된 문장", reason = "사용자 친화적이고 구체적인 개선 이유"
- **⚠️ 완벽한 문장은 corrections에 절대 포함하지 않음**: 맞춤법, 문법, 표현이 모두 올바른 문장은 corrections 배열에서 제외

📌 **개선 이유 작성 가이드** (사용자 친화적으로):
- ❌ 나쁜 예: "문법 오류 수정"
- ✅ 좋은 예: "문장이 더 자연스럽게 읽히도록 '~하여'로 연결했습니다"
- ❌ 나쁜 예: "어법 개선"  
- ✅ 좋은 예: "앞 문장과의 연결을 위해 '또한'을 추가하여 글의 흐름을 매끄럽게 했습니다"
- ❌ 나쁜 예: "표현 수정"
- ✅ 좋은 예: "더 구체적이고 임팩트 있는 표현으로 바꿔 인사담당자에게 강한 인상을 줄 수 있습니다"

📌 **예시 문장 작성 가이드** (missing_areas용):
- **⚠️ 중복 방지 원칙**: 자소서에 이미 언급된 구체적인 경험이나 내용과 겹치지 않는 **완전히 새로운** 예시만 작성
- **개인화된 새로운 예시 작성**: 자소서에 언급된 분야/관심사를 참고하되, 아직 언급되지 않은 **다른 경험**으로 예시 제공

**중복 방지 체크리스트:**
1. 자소서에 이미 언급된 구체적 경험(프로젝트, 활동, 학습 등)과 동일하거나 유사한 내용인지 확인
2. 이미 언급된 숫자(팀원 수, 기간, 성과 등)와 겹치는 내용인지 확인  
3. 동일한 기술/도구/방법론이 이미 언급되었는지 확인

**올바른 예시 작성법:**
- ❌ 자소서에 "Java 웹 프로젝트" 언급 → "Java 기반 웹 애플리케이션 개발..." (중복!)
- ✅ 자소서에 "Java 웹 프로젝트" 언급 → "알고리즘 스터디를 6개월간 운영하며 매주 문제 해결 방법을 공유했습니다" (새로운 경험)
- ❌ 자소서에 "5명 팀 프로젝트" 언급 → "5명의 팀원과 협업..." (숫자 중복!)  
- ✅ 자소서에 "5명 팀 프로젝트" 언급 → "개인 포트폴리오 프로젝트 3개를 독립적으로 완성하여 실력을 검증받았습니다" (다른 방식의 경험)

**분야별 새로운 예시 접근법:**
- 기술 분야: 프로젝트 경험 언급 시 → 스터디, 오픈소스 기여, 블로그 운영, 멘토링 등 다른 활동
- 협업 경험: 팀 프로젝트 언급 시 → 동아리 운영, 튜터링, 봉사활동 등 다른 협업  
- 학습 경험: 특정 기술 학습 언급 시 → 자격증 취득, 강의 수강, 세미나 참석 등 다른 학습

📌 문맥 고려 개선 예시:
- 이전 문장이 "팀워크의 중요성"을 언급했다면, 다음 문장은 "또한", "이와 더불어" 등으로 자연스럽게 연결
- 결론 부분에서는 "따라서", "이러한 경험을 바탕으로" 등의 표현 사용
- 시간 순서나 논리적 순서에 맞는 접속어와 표현 사용

📌 반환 형식 (JSON):
{{
  "analysis": {{
    "corrections": [
      {{
        "original": "제거되거나 개선할 원본 문장",
        "improved": "개선된 문장 (중복 제거의 경우 빈 문자열 '')",
        "reason": "중복 제거 이유 또는 문법/표현 개선 이유"
      }}
    ],
    "missing_areas": [
      {{
        "category": "부족한 영역 카테고리 (직무 역량 표현/성실성과 태도/리더십과 도전정신/결과 및 성과 중심 표현)",
        "description": "해당 영역이 부족한 이유와 중요성",
        "suggestions": [
          {{
            "title": "구체적인 개선 제안 제목",
            "content": "어떤 내용을 추가해야 하는지 상세 설명 (위 4가지 핵심 영역 기준으로 작성)",
            "example": "자소서에 이미 언급된 내용과 겹치지 않는 완전히 새로운 경험의 예시 문장 (따옴표나 '예를 들어' 등의 접두사 없이 완성된 문장만 작성)",
            "insertion_point": {{
              "target_sentence": "이 문장 뒤에 삽입하면 좋을 기존 문장",
              "reason": "해당 위치에 삽입하는 이유"
            }}
          }}
        ]
      }}
    ]
    
    ⚠️ **missing_areas가 없는 경우**: "missing_areas": [] (빈 배열로 반환)
  }}
}}

⚠️ 중요 규칙 (반드시 준수):
- 의미 유사 그룹에서는 반드시 1개 문장만 선택하고 나머지는 제거
- **중복 제거만**: improved = "" (빈 문자열), reason = "'[선택된 문장의 앞부분 15-20글자]...'와 의미상 중복 내용이라 삭제되었습니다"
- **실제 개선이 필요한 경우만**: improved = "개선된 문장", reason = "구체적 개선 이유 (맞춤법/문법/표현/연결성)"
- **🚫 절대 금지**: original과 improved가 완전히 동일한 corrections 생성 금지
- **🚫 절대 금지**: 이미 완벽한 문장을 corrections에 포함하는 것 금지
- **올바른 예시**: 10개 문장 중 2개만 중복 제거, 3개만 문법 개선이 필요하다면 → corrections는 5개만 생성

📌 중복 제거 이유 작성 가이드:
- 선택된 문장의 앞부분 15-20글자를 따옴표 안에 표시
- 예시: "'팀 프로젝트에서 리더 역할을...'와 의미상 중복 내용이라 삭제되었습니다"
- 사용자가 어떤 문장이 선택되었는지 비교할 수 있도록 도움

📌 corrections 올바른 예시 (사용자 친화적 이유 포함):
```json
[
  {{
    "original": "저는 끊임없이 학습하며 발전하는 개발자입니다.",
    "improved": "",
    "reason": "'항상 새로운 기술을 학습하고 실무에 적용하는...'와 의미상 중복 내용이라 삭제되었습니다"
  }},
  {{
    "original": "새로운 기술에 대한 호기심이 많고, 이를 실무에 적용하는 것을 좋아합니다.",
    "improved": "또한 새로운 기술에 대한 호기심이 많으며, 이를 실무에 적극적으로 적용하여 업무 효율성을 높이고자 합니다.",
    "reason": "앞 문장과의 연결을 위해 '또한'을 추가하고, '적극적으로', '업무 효율성' 등의 표현으로 더 구체적이고 전문적인 인상을 주도록 개선했습니다"
  }},
  {{
    "original": "저는 책임감이 강합니다. 맡은 업무를 완수합니다.",
    "improved": "저는 책임감이 강하여 맡은 업무를 끝까지 완수합니다.",
    "reason": "짧고 단조로운 두 문장을 '~하여'로 자연스럽게 연결하고, '끝까지'를 추가하여 의지가 더 강하게 느껴지도록 했습니다"
  }},
  {{
    "original": "팀워크를 중요하게 생각합니다.",
    "improved": "팀워크를 바탕으로 한 협업을 매우 중요하게 생각합니다.",
    "reason": "'협업'이라는 구체적 표현을 추가하고 '매우'로 강조하여 팀워크에 대한 진정성이 더 잘 드러나도록 했습니다"
  }}
]
```

📌 잘못된 예시 (생성하면 안 됨):
```json
[
  {{
    "original": "저는 성실하고 책임감 있는 사람입니다.",
    "improved": "저는 성실하고 책임감 있는 사람입니다.",
    "reason": "문법과 표현이 적절함"
  }},
  {{
    "original": "팀워크를 중요하게 생각합니다.",
    "improved": "팀워크를 중요하게 생각합니다.",
    "reason": "수정 불필요"
  }}
]
```
→ ⛔ **이런 경우는 corrections에 절대 포함하지 마세요!** 
→ ✅ **대신 완전히 무시하고 넘어가세요!**

📌 의미 유사 그룹들 (각 그룹에서 1개만 선택):
{similar_groups_text}

📌 단독 문장들 (문법/표현 개선 대상):
{single_sentences_text}
"""
        return prompt
    
    def call_gpt_analysis(self, grouped_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        GPT를 호출하여 자소서 분석 수행 (그룹화된 문장 기반)
        
        Args:
            grouped_data: 그룹화된 문장 데이터
            
        Returns:
            Dict[str, Any]: GPT 분석 결과
        """
        try:
            prompt = self.build_gpt_prompt(grouped_data)
            
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GPT_TEMPERATURE
            )
            
            result_text = response.choices[0].message.content
            
            # JSON 파싱 (markdown 코드 블록 제거)
            try:
                logger.info(f"GPT 원본 응답: {result_text[:500]}...")  # 응답 일부 로깅
                
                # markdown 코드 블록 제거
                clean_text = result_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]  # ```json 제거
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]  # ``` 제거
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]  # 끝의 ``` 제거
                clean_text = clean_text.strip()
                
                result = json.loads(clean_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"GPT 응답 JSON 파싱 실패: {str(e)}")
                logger.error(f"정리된 텍스트: {clean_text[:500]}...")
                return {
                    "analysis": {
                        "corrections": [],
                        "missing_areas": []
                    }
                }
                
        except Exception as e:
            logger.error(f"GPT 호출 중 오류 발생: {str(e)}")
            raise Exception(f"GPT 분석 중 오류가 발생했습니다: {str(e)}")
    
    def analyze_essay(self, original_sentences: List[str]) -> Dict[str, Any]:
        """
        자소서 전체 분석 프로세스 실행
        
        Args:
            original_sentences: 원본 문장들
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 1. 의미 유사 그룹핑
            groups = self.group_similar_sentences(original_sentences)
            
            # 2. 그룹화된 데이터 준비
            grouped_data = self.prepare_grouped_sentences_for_gpt(original_sentences, groups)
            
            # 3. GPT 분석 호출 (그룹별 최적 문장 선택 + 분석)
            gpt_result = self.call_gpt_analysis(grouped_data)
            
            # 4. 결과 구성
            result = {
                "original_sentences_count": len(original_sentences),
                "groups": groups,
                "grouped_data": grouped_data,
                **gpt_result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"자소서 분석 중 오류 발생: {str(e)}")
            raise Exception(f"자소서 분석 중 오류가 발생했습니다: {str(e)}") 