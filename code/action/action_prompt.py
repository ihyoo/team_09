# """
# Make VectorDB with huggingface
# """

# # !pip install langchain langchain-community langchain-openai
# # !pip install pymupdf
# # !pip install openai
# # !pip install chromadb

# import os
# import re
# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import Tool, initialize_agent
# from langchain.agents.agent_types import AgentType
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # ✅ 모델 초기화
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY2")

# news = NaverNewsAPI()
# llm = ChatOpenAI(temperature=0.3)


# # ✅ 후보자 정보
# candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]
# # PDF_FOLDER = "/home/ihy/yih/univ/2501/big_nlp/team_proj/team_09/team_09/data"
# PDF_FOLDER = "../../data"

# file_paths = {
#     name: [f"{PDF_FOLDER}20250604_대한민국_{name}_선거공약서.pdf"] * 2
#     for name in candidates
# }

# # ✅ 문서 로딩
# all_documents = []
# for name, paths in file_paths.items():
#     for path in paths:
#         loader = PyMuPDFLoader(path)
#         data = loader.load()
#         for d in data:
#             d.metadata["candidate"] = name
#             page = d.metadata.get("page", "?")
#             d.metadata["source"] = f"{os.path.basename(path)}:p{page}"
#         all_documents.extend(data)

# # ✅ 문서 분할
# splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=1000, chunk_overlap=200, encoding_name='cl100k_base'
# )
# documents = splitter.split_documents(all_documents)

# # ✅ 임베딩 및 벡터스토어 생성
# embedding_model = HuggingFaceEmbeddings(
#     model_name='jhgan/ko-sbert-nli',
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# vectorstore = Chroma.from_documents(documents, embedding=embedding_model, persist_directory="chroma_db")
# vectorstore.persist()

# # ✅ 후보별 Retriever
# retrievers = {
#     c: vectorstore.as_retriever(search_kwargs={"k": 6, "filter": {"candidate": c}})
#     for c in candidates
# }

# # ✅ 프롬프트
# policy_prompt = PromptTemplate(
#     input_variables=["summaries", "question"],
#     template="""
# 아래 문서는 대통령 후보의 공약이다. 질문에 대해 다음 요소를 포함하여 답변하라:
# - 정책의 목적
# - 구체적 수단 (시설, 제도, 법안 등)
# - 실행 대상 또는 지역
# - 문서상 등장한 구체적인 단어(용어)를 사용
# - 반드시 한국어로 답변할 것

# 문맥:
# {summaries}

# 질문:
# {question}

# 답변:
# """
# )

# compare_prompt_5way = PromptTemplate(
#     input_variables=["topic", "comparisons"],
#     template="""
# 다음은 '{topic}'에 대한 대통령 후보 공약 요약이다. 다음 기준에 따라 자세히 비교하라:

# 1. 정책의 목적 비교
# 2. 구체적 수단 비교
# 3. 실행 대상 또는 지역 비교
# 4. 문서상 등장한 구체적 용어 비교

# 아래 형식을 유지하고 문장을 요약하지 마시오. 반드시 문단 단위로 상세히 작성하고, 후보별 차이점을 구체적으로 명시하시오.

# 후보별 공약:
# {comparisons}

# 반드시 한국어로 작성하시오.
# """
# )

# translation_prompt = PromptTemplate(
#     input_variables=["english_text"],
#     template="다음 영어 텍스트를 자연스럽고 정확한 한국어로 번역하십시오:\n\n{english_text}\n\n번역:"
# )

# candidate_detect_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="다음 질문에서 언급된 대통령 후보의 이름을 정확히 추출하시오. 두 명 이상일 경우 모두 출력하시오.\n\n질문: {question}\n후보 이름:"
# )

# # ✅ 체인 구성
# qa_chains = {
#     name: RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         retriever=retrievers[name],
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": policy_prompt},
#         return_source_documents=True
#     ) for name in candidates
# }
# compare_chain_5way = LLMChain(llm=llm, prompt=compare_prompt_5way)
# translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
# candidate_chain = LLMChain(llm=llm, prompt=candidate_detect_prompt)

# # ✅ 유틸 함수
# def is_empty_or_irrelevant(answer: str) -> bool:
#     patterns = ["관련.*없", "언급되지 않았습니다", "포함되어 있지 않습니다", "등장하지 않습니다", "찾을 수 없습니다", "문서에.*없"]
#     return not answer.strip() or any(re.search(p, answer) for p in patterns)

# def is_english(text: str, threshold: float = 0.6) -> bool:
#     english_chars = re.findall(r'[a-zA-Z]', text)
#     total_chars = re.findall(r'\S', text)
#     return bool(total_chars) and len(english_chars) / len(total_chars) >= threshold

# def translate_if_needed(text: str) -> str:
#     if is_english(text):
#         result = translation_chain.invoke({"english_text": text})
#         return result["text"] if isinstance(result, dict) else result
#     return text

# # ✅ 외부 뉴스 요약 fallback (news_final_summary는 외부에서 정의)
# def fallback_policy_response(candidate: str, keyword: str) -> str:
#     try:
#         summary_result = news_final_summary(candidate, keyword)
#         return summary_result or f"{candidate} 후보의 '{keyword}' 관련 정책 정보를 외부에서 찾을 수 없습니다."
#     except Exception as e:
#         return f"{candidate} 후보의 '{keyword}' 관련 외부 뉴스 요약 중 오류가 발생했습니다: {e}"

# def format_candidate_policy(candidate: str, answer: str, is_external: bool) -> str:
#     source = "📰 외부 뉴스 요약 기반:" if is_external else "📄 PDF 기반 공약:"
#     return f"[{candidate} 후보]\n{source}\n{answer.strip()}"

# def format_final_comparison(topic: str, comparisons: list[str]) -> str:
#     return f"""✅ '{topic}'에 대한 후보별 공약 비교 분석 결과\n\n{chr(10).join(comparisons)}\n\n※ 일부 후보의 경우 PDF에 해당 정책이 없어 외부 뉴스 요약 정보가 사용되었습니다."""

# # ✅ 단일 후보 질의
# def run_candidate_policy_qa(input):
#     result = candidate_chain.invoke({"question": input})["text"]
#     target = next((c for c in candidates if c in result), None)
#     if not target:
#         return "후보 이름을 인식할 수 없습니다."

#     try:
#         answer = qa_chains[target].invoke({"question": input})["answer"]
#         is_external = is_empty_or_irrelevant(answer)
#         if is_external:
#             answer = fallback_policy_response(target, input)
#     except Exception:
#         is_external = True
#         answer = fallback_policy_response(target, input)

#     answer = translate_if_needed(answer)
#     return f"Final Answer:\n{format_candidate_policy(target, answer, is_external)}"

# # ✅ 다자 비교
# def run_policy_compare_all(input):
#     if "," in input:
#         split = [c.strip() for c in input.split(",")]
#         involved = [c for c in split if c in candidates]
#         keyword = next((k for k in split if k not in candidates), input)
#     else:
#         involved = candidates
#         keyword = input.strip()

#     comparisons = []
#     for cand in involved:
#         try:
#             raw_answer = qa_chains[cand].invoke({"question": keyword})["answer"]
#             is_external = is_empty_or_irrelevant(raw_answer)
#             answer = fallback_policy_response(cand, keyword) if is_external else raw_answer
#         except Exception:
#             is_external = True
#             answer = fallback_policy_response(cand, keyword)

#         answer = translate_if_needed(answer)
#         comparisons.append(format_candidate_policy(cand, answer, is_external))

#     result = compare_chain_5way.invoke({
#         "topic": keyword,
#         "comparisons": "\n\n".join(comparisons)
#     })["text"]
#     result = translate_if_needed(result)

#     return f"{format_final_comparison(keyword, comparisons)}\n\n✅ 최종 비교 분석\n{result}"

# # ✅ 툴 & 에이전트 설정
# react_tools = [
#     Tool(name="CandidatePolicyQA", func=run_candidate_policy_qa, description="질문에서 후보를 식별하고 해당 공약을 검색함."),
#     Tool(name="ComparePolicies", func=run_policy_compare_all, description="모든 후보의 공약을 비교하고 정책이 없으면 외부 함수 호출로 보완함.")
# ]

# agent = initialize_agent(
#     tools=react_tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={
#         "system_message": (
#             "당신은 반드시 한국어로 사고하고 응답하는 정책 분석 도우미입니다. "
#             "Thought, Action, Observation, Final Answer 형식을 사용하며, 모든 출력은 한국어여야 합니다. "
#             "**Final Answer는 반드시 정책 비교의 세부 내용을 포함하여 문단 형태로 자세히 작성하십시오.**"
#         )
#     },
#     handle_parsing_errors=True
# )



import os
import re
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ 환경 설정 및 모델 로딩
load_dotenv()
llm = ChatOpenAI(temperature=0.3)
embedding_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# ✅ 후보자 목록 및 Retriever
candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]
retrievers = {
    c: vectorstore.as_retriever(search_kwargs={"k": 6, "filter": {"candidate": c}})
    for c in candidates
}

# ✅ 프롬프트 구성
policy_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""..."""  # 기존 정책 요약 프롬프트 그대로 삽입
)
compare_prompt_5way = PromptTemplate(
    input_variables=["topic", "comparisons"],
    template="""..."""  # 기존 비교 프롬프트 그대로 삽입
)
translation_prompt = PromptTemplate(
    input_variables=["english_text"],
    template="다음 영어 텍스트를 자연스럽고 정확한 한국어로 번역하십시오:\n\n{english_text}\n\n번역:"
)
candidate_detect_prompt = PromptTemplate(
    input_variables=["question"],
    template="다음 질문에서 언급된 대통령 후보의 이름을 정확히 추출하시오. 두 명 이상일 경우 모두 출력하시오.\n\n질문: {question}\n후보 이름:"
)

# ✅ 체인 구성
qa_chains = {
    name: RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retrievers[name],
        chain_type="stuff",
        chain_type_kwargs={"prompt": policy_prompt},
        return_source_documents=True
    ) for name in candidates
}
compare_chain_5way = LLMChain(llm=llm, prompt=compare_prompt_5way)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
candidate_chain = LLMChain(llm=llm, prompt=candidate_detect_prompt)

# ✅ 유틸 함수들 (is_empty_or_irrelevant, translate_if_needed 등은 그대로 유지)

# ✅ 질의 함수 (run_candidate_policy_qa, run_policy_compare_all 등은 그대로 유지)

# ✅ ReAct Agent 설정
react_tools = [
    Tool(name="CandidatePolicyQA", func=run_candidate_policy_qa, description="질문에서 후보를 식별하고 해당 공약을 검색함."),
    Tool(name="ComparePolicies", func=run_policy_compare_all, description="모든 후보의 공약을 비교하고 정책이 없으면 외부 함수 호출로 보완함.")
]

agent = initialize_agent(
    tools=react_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": (
            "당신은 반드시 한국어로 사고하고 응답하는 정책 분석 도우미입니다. "
            "Thought, Action, Observation, Final Answer 형식을 사용하며, 모든 출력은 한국어여야 합니다. "
            "**Final Answer는 반드시 정책 비교의 세부 내용을 포함하여 문단 형태로 자세히 작성하십시오.**"
        )
    },
    handle_parsing_errors=True
)

# ✅ 예시 실행
# agent.run("이재명 후보의 청년 정책은?")