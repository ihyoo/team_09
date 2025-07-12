
import os
import re
import openai
from dotenv import load_dotenv
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
import gradio as gr

from embedding.document_loader import process_documents

from functions import *
from naver import *
from prompts import *

candidates = ["이재명", "김문수", "이준석", "권영국", "송진호"]

retrievers = process_documents()


def translate_if_needed(text: str) -> str:
    if is_english(text):
        result = translation_chain.invoke({"english_text": text})
        return result["text"] if isinstance(result, dict) else result
    return text

#질의와 Agent응답 결과간의 관련성 체크
def is_llm_irrelevant(question: str, answer: str) -> bool:

    try:
        # print(f"[관련성 판단 호출] Q: {question} / A: {answer}")
        result = relevance_chain.invoke({"question": question, "answer": answer})
        result_text = result["text"].strip() if isinstance(result, dict) else result.strip()
        print(f"관련성 판단 결과: {result_text}")
        return "관련 없음" in result_text
    except Exception as e:
        print(f"❌ 관련성 판단 실패: {e}")
        return False


# ✅ 후보별 요약 수집 및 비교표 정보 생성
def summarize_all_candidates(keywords: list[str], question: str) -> tuple[str, list[dict]]:
    summaries = []
    comparison_rows = []
    for cand in candidates:
        try:
            full_summary = []
            for kw in keywords:
                result = qa_chains[cand].invoke({"question": kw})["answer"].strip()
                full_summary.append(f"- {kw}: {result}")

            joined_summary = "\n".join(full_summary)
            summaries.append(f"[{cand} 후보]\n{joined_summary}")

            # 상세 분해 추출
            extract = detailed_policy_chain.invoke({"question": question, "summary": joined_summary})["text"]
            extract_lines = extract.split("\n")
            row = {
                "후보": cand,
                "핵심 공약": extract_lines[0].replace("핵심 공약:", "").strip() if len(extract_lines) > 0 else "",
                "실현 방식": extract_lines[1].replace("실현 방식:", "").strip() if len(extract_lines) > 1 else "",
                "강점": extract_lines[2].replace("강점:", "").strip() if len(extract_lines) > 2 else ""
            }
            comparison_rows.append(row)

        except Exception as e:
            summaries.append(f"[{cand} 후보] 요약 실패: {e}")
            comparison_rows.append({"후보": cand, "핵심 공약": "요약 실패", "실현 방식": "", "강점": ""})

    return "\n\n".join(summaries), comparison_rows

# ✅ 단일 후보 공약 질의 실행 함수
def run_candidate_policy_qa(input: str) -> str:
    print("#### 단일 후보 공약 질의 Tool 활성화....")
    print("#### 후보자 이름 추출 중....")
    result = candidate_chain.invoke({"question": input})["text"]
    target = next((c for c in candidates if c in result), None)
    if not target:
        return "Final Answer: 후보 이름을 인식할 수 없습니다."

    print(f"#### {target} 후보자 공약집 검색 중....")
    result_dict = qa_chains[target].invoke({"question": input})
    answer = result_dict["answer"].strip()
    sources = result_dict.get("source_documents", [])

    print("#### 관련성 체크 중....")
    if is_empty_or_irrelevant(answer) or is_llm_irrelevant(input, answer) or len(sources) <= 1:
        return f"Final Answer: 공약 없음"
    else:
        print("#### 단일 후보 공약 RAG 실행 완료 ####")

        print("#### 사후 포맷팅 진행 ####")
        print("#"*50)
        answer = translate_if_needed(answer)
        final_text = single_candidate_formatter_chain.invoke({
                "question": input,
                "summaries": answer,
                "sources": sources,
            })["text"]

        # #✅ 전역 또는 외부에서 접근 가능하도록 저장 (예: 전역 dict)
        # global last_observation_output
        # last_observation_output = final_text  # ✅ 이 변수에 저장됨

        return f"Final Answer:\n{final_text}"



# ✅ 다자 공약 비교 실행 함수
def run_policy_compare_all(input):
    print("\n#### 후보별 공약 비교 Tool 활성화....")
    print("##### 후보자 이름 추출 중....")
    if "," in input:
        split = [c.strip() for c in input.split(",")]
        involved = [c for c in split if c in candidates]
        keyword = next((k for k in split if k not in candidates), input)
    else:
        involved = candidates
        keyword = input.strip()

    print(f"#### 후보자 이름 : {involved}")
    #RAG 실행
    comparisons = []
    for cand in involved:
        print(f"#### {cand} 후보자 공약집 검색 중....")
        result_dict = qa_chains[cand].invoke({"question": keyword})
        answer = result_dict["answer"]
        sources = result_dict.get("source_documents", [])

        if is_empty_or_irrelevant(answer) or is_llm_irrelevant(input, answer) or len(sources) <= 1:
            answer = "공약 없음"

        answer = translate_if_needed(answer)
        answer = format_candidate_policy(cand, answer)

        comparisons.append(answer)

    print(f"#### 공약 비교 중....")
    result = compare_chain.invoke({
        "topic": keyword,
        "comparisons": "\n\n".join(comparisons)
    })["text"]
    result = translate_if_needed(result)

    print("#### 후보별 공약 비교 완료 ####")
    print("#"*50)

    # ✅ 최종 텍스트 구성
    final_text = f"{format_final_comparison(keyword, comparisons)}\n\n✅ 최종 비교 분석\n{result}"

    # if "last_observation_output" not in globals():
    #     #✅ 전역 또는 외부에서 접근 가능하도록 저장 (예: 전역 dict)
    #     global last_observation_output
    #     last_observation_output = final_text  # ✅ 이 변수에 저장됨        
    # else :
    #     last_observation_output = ""

    return f"Final Answer:\n{final_text}"


# ✅ 사용자 맞춤 추천 실행 함수
def run_user_profile_recommendation(question: str) -> str:

    print("\n#### 사용자 컨텍스트 별 질의 응답 Tool 활성화....")
    print("##### 사용자 프로파일 추출 중....")

    # 1. 사용자 프로파일 추출
    profile_result = extract_user_profile_chain.invoke({"question": question})["text"]


    # 2. 관심 키워드 추출
    keywords = re.findall(r'관심정책 키워드\s*\(.*?\):\s*(.*)', profile_result)
    keyword_list = keywords[0].split(",") if keywords else []

    print("##### 후보별 공약 요약 중....")

    # 3. 후보별 공약 요약
    summaries, comparison_rows = summarize_all_candidates([kw.strip() for kw in keyword_list], question)

    # 4. 추천 수행
    recommendation = recommend_chain.invoke({
        "question": question,
        "profile": profile_result,
        "summaries": summaries
    })["text"]

    question = translate_if_needed(question)

    # ✅ 최종 텍스트 구성
    final_text = format_recommendation_output(question, comparison_rows, recommendation)

    print("#### 사용자 컨텍스트 별 질의 응답 완료 ####")
    print("#"*50)

    # if "last_observation_output" not in globals():
    #     #✅ 전역 또는 외부에서 접근 가능하도록 저장 (예: 전역 dict)
    #     global last_observation_output
    #     last_observation_output = final_text  # ✅ 이 변수에 저장됨        
    # else :
    #     last_observation_output = ""

    return "Final Answer:\n" + final_text



# ✅ 실시간 뉴스 검색
def run_policy_news_issue(input: str) -> str:
    print("\n#### 실시간 여론 반응 분석 Tool 활성화....")
    print("#### 후보자 이름 추출 중....")
    result = candidate_chain.invoke({"question": input})["text"]
    candidate = next((c for c in candidates if c in result), None)
    if not candidate:
        return "Final Answer: 후보 이름을 인식할 수 없습니다."


    # 정책 키워드 추출 (후보명 제외 나머지)
    keyword = input.replace(candidate, "").strip(" ,")
    if not keyword:
        return "Final Answer: 정책 주제(키워드)를 포함해 질문해 주세요."

    print(f'#### {candidate} 후보의 {keyword} 관련 정책 뉴스 검색 중...')
    try:
        news_summary = news_final_summary(candidate, keyword)

        # ✅ 최종 텍스트 구성
        final_text = news_issue_summary_chain.invoke({
                    "candidate": candidate,
                    "keyword": keyword,
                    "news_summary": news_summary
                })["text"]


        print("#### 실시간 여론 반응 분석 완료 ####")
        print("#"*50)

        # if "last_observation_output" not in globals():
        #     #✅ 전역 또는 외부에서 접근 가능하도록 저장 (예: 전역 dict)
        #     global last_observation_output
        #     last_observation_output = final_text  # ✅ 이 변수에 저장됨        
        # else :
        #     last_observation_output = ""

        return final_text or f"{candidate} 후보의 '{keyword}' 관련 정책 정보를 외부에서 찾을 수 없습니다."
    except Exception as e:
        return f"{candidate} 후보의 '{keyword}' 관련 외부 뉴스 요약 중 오류가 발생했습니다: {e}"


# ✅ 툴 & 에이전트 설정
tool_qa = Tool(name="CandidatePolicyQA", func=run_candidate_policy_qa, description="후보 이름과 주제를 기반으로 공약을 PDF에서 찾고, 공약 없음이면 바로 종료한다.")
tool_compare = Tool(name="ComparePolicies", func=run_policy_compare_all, description="복수 후보 간 특정 주제에 대한 공약을 비교합니다.")
tool_recommend = Tool(name="RecommendCandidateNatural", func=run_user_profile_recommendation, description="사용자 질문에서 상황을 추출하고 적합한 후보를 추천합니다.")
tool_news = Tool(name="PolicyNewsIssueCheck",func=run_policy_news_issue,description="후보와 정책 키워드에 대해 최근 뉴스나 논란, 실제 이행 상황 등 사회적 이슈를 요약합니다."
)

react_tools = [tool_qa, tool_compare, tool_recommend,tool_news]

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2)

agent = initialize_agent(
    tools=react_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # verbose=True면 Thought/Action 로그 출력됨
    agent_kwargs={
        "system_message": (
            "당신은 반드시 한국어로 사고하고 응답하는 정책 분석 도우미입니다. "
            "Thought, Action, Observation, Final Answer 형식을 사용하며, "
            "**Final Answer로 시작하는 응답이 나오면 체인을 반드시 종료하십시오.**"
        )
    },
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

translation_prompt = PromptTemplate(
    input_variables=["english_text"],
    template="다음 영어 텍스트를 자연스럽고 정확한 한국어로 번역하십시오:\n\n{english_text}\n\n번역:"
)

#후보자 이름 추출
candidate_detect_prompt = PromptTemplate(
    input_variables=["question"],
    template="다음 질문에서 언급된 대통령 후보의 이름을 정확히 추출하시오. 두 명 이상일 경우 모두 출력하시오.\n\n질문: {question}\n후보 이름:"
)


# ✅ 후보별 QA 체인 초기화
qa_chains = {
    name: RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retrievers[name],
        chain_type="stuff",
        chain_type_kwargs={"prompt": single_candidate_policy_prompt},
        return_source_documents=True
    )
    for name in candidates
}

candidate_chain = LLMChain(llm=llm, prompt=candidate_detect_prompt) #후보자 이름 추출
compare_chain = LLMChain(llm=llm, prompt=multi_candidate_comparison_prompt)# 공약 비교 체인
extract_user_profile_chain = LLMChain(llm=llm, prompt=user_profile_extraction_prompt) #사용자 프로필 추출
recommend_chain = LLMChain(llm=llm, prompt=candidate_recommendation_prompt) #후보자 추천
relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt) # 질의와 응답간 관련성 체크
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)# 한국어로 번역
detailed_policy_chain = LLMChain(llm=llm, prompt=policy_element_extraction_prompt)
news_issue_summary_chain = LLMChain(llm=llm, prompt=news_issue_summary_prompt) #뉴스/논란
single_candidate_formatter_chain = LLMChain(llm=llm, prompt=fomatting_single_candidate_prompt) #후보자 공약 출력 포맷팅

def action_query(query) :

    response = agent.invoke({"input": query})
    final_answer = response["output"]

    sub_response = response["intermediate_steps"]

    print(f"final_answer - {final_answer}")
    print(f"sub_response - {sub_response}")

    for action, observation in sub_response:
        if isinstance(observation, str) and observation.startswith("Final Answer:"):
            sub_answer = observation  # 전체 Final Answer 텍스트 반환
            # print(f"observation - {observation}")
        elif hasattr(action, "tool") and action.tool == "PolicyNewsIssueCheck":
            if "[뉴스/이행 및 논란 요약]" in observation:
                # 필요하면 이 부분에서 "[뉴스/이행 및 논란 요약]" 이후만 추출 가능
                sub_answer = observation
            else :
                sub_answer = ""
        else :
            sub_answer = ""

    if is_english(final_answer):
        translated = translation_chain.invoke({"english_text": final_answer})
        if isinstance(translated, dict):
            final_answer = translated["text"].strip()
        else:
            final_answer = translated

    if is_english(sub_answer):
        translated = translation_chain.invoke({"english_text": sub_answer})
        if isinstance(translated, dict):
            sub_answer = translated["text"].strip()
        else:
            sub_answer = translated

    return final_answer, sub_answer

def action_query_wrapper(question):
    result, sub_result = action_query(question)
    return result, sub_result

def reset():
    return "", "", ""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🧭 유권자 나침반 질의-응답")
    gr.Markdown("### 대통령 후보 공약 기반 AI Agent")
    gr.Markdown("사용 방법 : 질문을 입력하고 '질문하기'를 누르세요. 초기화 버튼으로 모두 비울 수 있습니다.")

    with gr.Row():
        question_input = gr.Textbox(label="질문 입력", placeholder="예: 이재명 후보의 조선해양 정책에 대해 알려줘.", lines=2)
        ask_button = gr.Button("❓ 질문하기", variant="primary")
        reset_button = gr.Button("🔄 초기화", variant="secondary")

    result_output = gr.Textbox(label="📥 한 줄 요약 및 결과", lines=4, interactive=False)
    state_output = gr.Textbox(label="🧠 상세 요약", lines=2, interactive=False)

    ask_button.click(fn=action_query_wrapper, inputs=question_input, outputs=[result_output, state_output])
    reset_button.click(fn=reset, outputs=[question_input, result_output, state_output])
    
demo.launch()

# result = action_query("이재명 후보의 조선해양 정책에 대해 알려줘.")
# print(result)
# print(last_observation_output)