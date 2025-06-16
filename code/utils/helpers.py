import re

def is_empty_or_irrelevant(answer: str) -> bool:
    """
    답변이 비어있거나 정책적 관련성이 없는 표현(예: '관련 없음', '언급되지 않았습니다' 등)이 포함되어 있는지 판정
    """
    patterns = [
        r"관련.*없",
        r"언급되지 않았습니다",
        r"포함되어 있지 않습니다",
        r"등장하지 않습니다",
        r"찾을 수 없습니다",
        r"문서에.*없"
    ]
    return not answer.strip() or any(re.search(p, answer) for p in patterns)

def is_english(text: str, threshold: float = 0.6) -> bool:
    """
    입력 텍스트가 영어 위주인지 판정 (영문자 비율이 threshold 이상이면 True)
    """
    english_chars = re.findall(r'[a-zA-Z]', text)
    total_chars = re.findall(r'\S', text)
    return bool(total_chars) and len(english_chars) / len(total_chars) >= threshold

def translate_if_needed(text: str, translation_chain=None) -> str:
    """
    텍스트가 영어일 경우 translation_chain을 이용해 한국어로 번역, 아니면 원문 반환
    """
    if is_english(text):
        if translation_chain is None:
            raise ValueError("translation_chain이 필요합니다.")
        result = translation_chain.invoke({"english_text": text})
        return result["text"] if isinstance(result, dict) else result
    return text

def format_candidate_policy(candidate: str, answer: str) -> str:
    """
    후보자 이름과 PDF 기반 공약 답변을 보기 좋게 포맷팅
    """
    return f"[{candidate} 후보]\n📄 PDF 기반 공약: \n{answer.strip()}"
