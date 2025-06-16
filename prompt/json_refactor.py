import json
import re

def fix_comma_in_questions_block(raw_json: str):
    # 모든 질문 블록 사이에 쉼표가 누락된 경우를 처리
    # "질문\d+": { ... } 블록을 기준으로 잡음
    blocks = re.findall(r'("질문\d+"\s*:\s*\{[\s\S]*?\})', raw_json)

    # 각 블록 뒤에 쉼표 추가, 마지막 블록은 빼기
    if len(blocks) <= 1:
        return raw_json  # 질문이 하나면 상관없음

    joined = ',\n'.join(blocks)
    final_json = f"{{\n{joined}\n}}"
    return final_json


def fix_commas_in_question_array(raw_json: str) -> str:
    """
    'questions': [ {...} {...} {...} ] 와 같은 구조에서
    JSON 배열 요소 사이에 쉼표가 빠졌을 때 자동 보정합니다.
    """
    # 배열 내부의 각 객체 블록 추출
    question_blocks = re.findall(r'(\{\s*"question"\s*:\s*"[\s\S]*?\})', raw_json)

    if len(question_blocks) <= 1:
        return raw_json  # 하나일 경우는 문제 없음

    # 쉼표로 연결
    fixed_block = ',\n'.join(question_blocks)
    fixed_json = re.sub(
        r'"questions"\s*:\s*\[\s*[\s\S]*?\s*\]',
        f'"questions": [\n{fixed_block}\n]',
        raw_json
    )

    return fixed_json



def extract_and_fix_gpt_json(content: str):
    content = content.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{[\s\S]+\}", content)
    if not match:
        return {"error": "JSON 응답을 찾을 수 없습니다.", "raw": content}

    raw_json = match.group()

    # 1. 쉼표 보정
    if '"questions": [' in raw_json:
        fixed_json = fix_commas_in_question_array(raw_json)
    else:
        fixed_json = fix_comma_in_questions_block(raw_json)

    # 2. 파싱 시도
    try:
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        return {
            "error": f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno}",
            "pos": e.pos,
            "raw_fragment": fixed_json[e.pos-30:e.pos+30],
            "raw": fixed_json
        }

def merge_numbered_fields(data: dict, remove_original: bool = True) -> dict:
    """
    numbered 항목들을 '/'로 병합해 하나의 필드로 만들되,
    해당 prefix의 값이 전부 None일 경우는 병합 결과도 None으로 설정.
    """
    target_prefixes = [
        "bad_description", "bad_summary",
        "good_description", "good_summary",
        "cause", "state",
        "improvment", "solution"
    ]

    for prefix in target_prefixes:
        merged_items = []

        # 접미 숫자 포함된 키 추출 및 정렬
        numbered_keys = sorted(
            [k for k in data if re.fullmatch(f"{prefix}[0-9]+", k)],
            key=lambda x: int(re.search(r"\d+", x).group())
        )

        for key in numbered_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                merged_items.append(value.strip())

        # 값이 하나라도 있으면 병합 / 모두 None이면 None
        if merged_items:
            data[prefix] = "/".join(merged_items)
        elif numbered_keys:
            data[prefix] = None

        # 기존 키 제거
        if remove_original:
            for key in numbered_keys:
                data.pop(key, None)
    return data



