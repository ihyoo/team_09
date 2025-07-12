import os
from typing import List, Dict
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ✅ 프로젝트 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ✅ 데이터 및 벡터 DB 경로 설정
DATA_DIR = os.path.join(BASE_DIR, "raw_data")  # PDF 원본 위치
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorDB")
VECTOR_DB_NAME = "candidate_250712"
PERSIST_PATH = os.path.join(VECTOR_DB_DIR, VECTOR_DB_NAME)
os.makedirs(PERSIST_PATH, exist_ok=True)

# ✅ 후보자 정의
candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]

# ✅ 파일 경로 매핑
def get_file_paths() -> Dict[str, str]:
    """후보자 이름 기반 PDF 경로 찾기"""
    file_ls = os.listdir(DATA_DIR)
    file_paths = {}
    for name in candidates:
        matched_files = [f for f in file_ls if name in f]
        if not matched_files:
            raise FileNotFoundError(f"❌ {name} 후보의 공약 PDF가 존재하지 않습니다.")
        file_paths[name] = os.path.join(DATA_DIR, matched_files[0])
    return file_paths

# ✅ 문서 로딩
def load_documents(file_paths: Dict[str, str]) -> List[Document]:
    """PDF 문서 로딩 및 메타데이터 설정"""
    all_documents = []
    for name, path in file_paths.items():
        loader = PyMuPDFLoader(path)
        data = loader.load()
        for d in data:
            d.metadata.update({
                "candidate": name,
                "source": f"{os.path.basename(path)}:p{d.metadata.get('page', '?')}"
            })
        all_documents.extend(data)
    return all_documents

# ✅ 문서 분할
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200, encoding_name='cl100k_base'
    )
    return splitter.split_documents(documents)

# ✅ 임베딩 모델
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# ✅ 벡터스토어 생성
def create_vector_store(documents: List[Document]) -> Chroma:
    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=PERSIST_PATH
    )
    vectorstore.persist()
    return vectorstore

# ✅ 후보별 Retriever 생성
def create_retrievers(vectorstore: Chroma) -> Dict[str, Chroma]:
    return {
        c: vectorstore.as_retriever(search_kwargs={"k": 6, "filter": {"candidate": c}})
        for c in candidates
    }

# ✅ 전체 문서 처리 파이프라인
def process_documents() -> Dict[str, Chroma]:
    file_paths = get_file_paths()
    raw_documents = load_documents(file_paths)
    split_docs = split_documents(raw_documents)
    vectorstore = create_vector_store(split_docs)
    retrievers = create_retrievers(vectorstore)
    return retrievers

# ✅ 실행 시점
if __name__ == "__main__":
    retrievers = process_documents()
    print(f"✅ 벡터 DB 저장 위치: {PERSIST_PATH}")
    print("✅ 문서 처리 및 벡터스토어 구축 완료")
