"""
Load VectorDB from huggingface embedding
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vdb():
    # ✅ 디렉토리 설정
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorDB")
    VECTOR_DB_NAME = "candidate"
    PERSIST_PATH = os.path.join(VECTOR_DB_DIR, VECTOR_DB_NAME)

    # ✅ 후보자 정의
    candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]

    # ✅ 임베딩 모델 설정 (저장 시와 동일해야 함)
    embedding_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # ✅ 벡터 DB 로드
    vectorstore = Chroma(
        persist_directory=PERSIST_PATH,
        embedding_function=embedding_model
    )

    # ✅ 후보자별 retriever 생성
    retrievers = {
        c: vectorstore.as_retriever(search_kwargs={"k": 6, "filter": {"candidate": c}})
        for c in candidates
    }

    print(f"✅ VectorDB 로드 완료: {PERSIST_PATH}")
    return retrievers

# 사용 예시:
if __name__ == '__main__':
    retrievers = load_vdb()
