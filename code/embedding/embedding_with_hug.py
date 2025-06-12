"""
Make VectorDB with huggingface
"""

import os
# import fitz
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def make_vdb() :
    # ✅ code 기준으로 상위 team_09 디렉토리 경로 설정
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ✅ 데이터 및 벡터 DB 경로 설정
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorDB")
    VECTOR_DB_NAME = "candidate"
    PERSIST_PATH = os.path.join(VECTOR_DB_DIR, VECTOR_DB_NAME)
    os.makedirs(PERSIST_PATH, exist_ok=True)
    
    print(f"BASE_DIR - {BASE_DIR}")

    # # ✅ 벡터 DB 저장 경로
    # VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorDB")
    # VECTOR_DB_NAME = "candidate_vdb"
    # PERSIST_PATH = os.path.join(VECTOR_DB_DIR, VECTOR_DB_NAME)
    # os.makedirs(PERSIST_PATH, exist_ok=True)

    # ✅ PDF 데이터 경로
    DATA_DIR = os.path.join(BASE_DIR, "data")
    print(f"DATA_DIR - {DATA_DIR}")

    # ✅ 후보자 정보
    candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]
    
    file_ls = os.listdir(DATA_DIR)    

    file_paths = {
        name: [os.path.join(DATA_DIR, [x for x in file_ls if name in x][0])] * 2
        for name in candidates
    }

    print(file_paths)

    # ✅ 문서 로딩
    all_documents = []
    for name, paths in file_paths.items():
        for path in paths:
            loader = PyMuPDFLoader(path)
            data = loader.load()
            for d in data:
                d.metadata["candidate"] = name
                d.metadata["source"] = f"{os.path.basename(path)}:p{d.metadata.get('page', '?')}"
            all_documents.extend(data)    

    # ✅ 문서 분할
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )
    documents = splitter.split_documents(all_documents)

    # ✅ 임베딩 모델 설정
    embedding_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # ✅ 벡터 DB 생성 및 저장
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embedding_model,
        persist_directory=PERSIST_PATH
    )
    vectorstore.persist()

    print(f"✅ VectorDB 저장 완료: {PERSIST_PATH}")


if __name__ == '__main__':
    make_vdb()