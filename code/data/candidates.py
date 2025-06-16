# ✅ 후보자 정보 관리 모듈

# 1. 후보자 리스트
candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]

# 2. PDF 파일 경로 설정
PDF_FOLDER = "/content/"
file_paths = {
    name: [f"{PDF_FOLDER}20250604_대한민국_{name}_선거공약서.pdf"] * 2
    for name in candidates
}

# 3. 문서 메타데이터 기본 구조 (참조용)
DEFAULT_METADATA = {
    "candidate": "",
    "source": "",
    "page": 0
}

# 4. 후보자 색인 정보 (검색 최적화)
CANDIDATE_INDEX_MAP = {
    idx: {"name": name, "file": file_paths[name][0]}
    for idx, name in enumerate(candidates)
}
