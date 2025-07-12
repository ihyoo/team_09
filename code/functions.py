import os
import re
from difflib import SequenceMatcher
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ 유틸 함수
def is_empty_or_irrelevant(answer: str) -> bool:
    patterns = ["관련.*없", "언급되지 않았습니다", "포함되어 있지 않습니다", "등장하지 않습니다", "찾을 수 없습니다", "문서에.*없"]
    return not answer.strip() or any(re.search(p, answer) for p in patterns)

def is_english(text: str, threshold: float = 0.6) -> bool:
    english_chars = re.findall(r'[a-zA-Z]', text)
    total_chars = re.findall(r'\S', text)
    return bool(total_chars) and len(english_chars) / len(total_chars) >= threshold

def format_candidate_policy(candidate: str, answer: str) -> str:
    return f"[{candidate} 후보]\n📄 PDF 기반 공약: \n{answer.strip()}"


# ✅ 불릿 형식 비교 요약 생성
def build_bullet_summary(rows: list[dict]) -> str:
    lines = ["[후보별 공약 요약]"]
    for row in rows:
        lines.append(f"- {row['후보']} 후보:")
        lines.append(f"  • 핵심 공약: {row['핵심 공약']}")
        lines.append(f"  • 실현 방식: {row['실현 방식']}")
        lines.append(f"  • 강점: {row['강점']}")
    return "\n".join(lines)


# ✅ 출력 포맷 조립
def format_recommendation_output(question: str, comparison_rows: list[dict], recommendation_text: str) -> str:
    bullets = build_bullet_summary(comparison_rows)
    return f"""[질문]
{question}

{bullets}

[추천 요약]
{recommendation_text}
"""


def format_final_comparison(topic: str, comparisons: list[str]) -> str:
    return f"""✅ '{topic}'에 대한 후보별 공약 비교 분석 결과\n\n{chr(10).join(comparisons)}"""
