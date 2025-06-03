"""
Make VectorDB
"""

import os, time, json
import fitz  # PyMuPDF
from pathlib import Path
import chromadb
from openai import OpenAI
import unicodedata
from dotenv import load_dotenv

import sys
# 현재 파일의 절대 경로에서 5단계 상위로 이동 (team_proj까지)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from log_utils.logger_config import get_logger
logger = get_logger("embeddings")


def normalize_korean(text):
    return unicodedata.normalize("NFC", text.strip().replace(" ", "").replace("\u200b", ""))

def clean_category(category_raw):
    # 다양한 구분자 제거 및 첫 번째만 추출
    for sep in ["·", "ㆍ", "/", "|", ";", "‧"]:
        category_raw = category_raw.replace(sep, ",")
    parts = [p.strip() for p in category_raw.split(",") if p.strip()]
    first = parts[0] if parts else "모두"
    return "모두" if first == "기타" else first

def infer_party_from_question(question):
    party_synonyms = {
        "더불어민주당": ["더민주", "민주당", "민주"],
        "국민의힘": ["국힘", "국민의 힘", "국힘당"],
        "개혁신당": ["개신", "개혁"],
        "민주노동당": ["민노당", "민노"],
        "무소속": ["무소속", "독립", "무당"]
    }
    for formal, synonyms in party_synonyms.items():
        for word in synonyms + [formal]:
            if word in question:
                return formal
    return "모두"

# ✅ 텍스트 임베딩
def get_gpt_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.info(f"Embedding error: {e}")
        return None

# ✅ GPT 호출
def get_completion(system_msg, user_prompt, model="gpt-4", retry=3):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}]
    for i in range(retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.info(f"Retry {i+1} error: {e}")
            time.sleep(3)
    return None

# ✅ 메타데이터 분류
def classify_policy(text, candidate, party):
    prompt = f"""
다음 정책 내용을 보고 아래 정보를 JSON으로 추출해줘.
⚠️ 반드시 JSON 형식을 지켜줘. 이중 따옴표(\"), 줄바꿈 없음, 응답 외 텍스트 없이.

\"{text[:1000]}\"

- category: 경제, 복지, 교육, 주거, 부동산, 노동, 보건, 의료, 과학기술, 산업, 외교, 안보, 환경, 기후, 청년, 정책, 여성, 가족, 정치, 행정, 개혁, 디지털, 미디어, 기타 중 하나
- target_age: 청년, 중장년, 노년, 모두
- target_gender: 남성, 여성, 모두
- region: 서울, 전국 등
- party: {party} ← 이 값 그대로 넣어줘

형식:
{{"category":"...","target_age":"...","target_gender":"...","region":"...","party":"..."}}
"""
    result = get_completion("너는 정책 분석 전문가야.", prompt)
    try:
        return json.loads(result)
    except:
        return {"category": "기타", "target_age": "모두", "target_gender": "모두", "region": "전국", "party": party}

# ✅ PDF 처리 함수
def process_pdf_and_store(pdf_path):
    candidate_party = {
        "이재명": "더불어민주당",
        "김문수": "국민의힘",
        "이준석": "개혁신당",
        "권영국": "민주노동당",
        "황교안": "무소속",
        "송진호": "무소속"
    }
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).stem  # ex: 20250604_대한민국_이재명_선거공약서
    parts = filename.split("_")     # → ['20250604', '대한민국', '이재명', '선거공약서']
    parts = [x.strip().replace(" ","") for x in parts]

    if len(parts) < 4:
        logger.info(f"⚠️ 파일명 구성이 잘못됨: {filename}")
        return

    file_type = parts[-1]  # 마지막 요소 → '정당정책' or '선거공약서'
    # file_type = normalize_korean(file_type)
    logger.info(f"file_type - {file_type}")

    if file_type == "정당정책":
        party = parts[2].strip().replace(" ","")
        # party = normalize_korean(party)
        candidate = ""
        policy_type = "party"
    elif file_type == "선거공약서":
        candidate = parts[2].strip().replace(" ","")
        candidate = normalize_korean(candidate)
        party = candidate_party.get(candidate, "무소속")
        logger.info(f"candidate - {candidate}")
        logger.info(f"party - {party}")
        policy_type = "candidate"
    else:
        logger.info(f"⚠️ 알 수 없는 파일 유형: {filename}")
        return

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) < 100:
            continue

        logger.info(f"🔹 저장 시도: {candidate or party}_{i}")
        meta = classify_policy(text, candidate, party)
        meta["category"] = clean_category(meta.get("category", "모두"))
        embedding = get_gpt_embedding(text)

        if embedding:
            log_meta_data = {
                "candidate": candidate,
                "party": party,
                "policy_type": policy_type,
                "category": meta["category"],
                "target_age": meta["target_age"],
                "target_gender": meta["target_gender"],
                "region": meta["region"]
            }
            logger.info(f"→ 저장 메타: {log_meta_data}")
            
            collection.add(
                ids=[f"{candidate or party}_{i}"],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{
                    "candidate": candidate,
                    "party": party,
                    "policy_type": policy_type,
                    "category": meta["category"],
                    "target_age": meta["target_age"],
                    "target_gender": meta["target_gender"],
                    "region": meta["region"]
                }]
            )
    doc.close()


if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    
    
    PDF_FOLDER = "/home/ihy/yih/univ/2501/big_nlp/team_proj/team_09/data"
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    chroma_client = chromadb.PersistentClient(path="/home/ihy/yih/univ/2501/big_nlp/team_proj/team_09/vectorDB/chroma_policy_db")
    collection = chroma_client.get_or_create_collection(name="policy_collection")

    # ✅ 전체 PDF 실행
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    print(f"pdf_files - {pdf_files}")
    for pdf in pdf_files:
        process_pdf_and_store(str(pdf))
