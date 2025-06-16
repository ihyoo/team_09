import re
from typing import List, Dict
from langchain.chains import LLMChain
from langchain.schema import Document
from data.candidates import candidates
from llm.chains import (
    qa_chains,
    candidate_chain,
    compare_chain,
    extract_user_profile_chain,
    recommend_chain,
    detailed_policy_chain,
    translation_chain,
    news_issue_summary_chain,
    single_candidate_formatter_chain
)
from utils.helpers import (
    is_empty_or_irrelevant,
    is_english,
    translate_if_needed,
    format_candidate_policy
)

# 전역 상태 관리 변수
last_observation_output = None

def run_candidate_policy_qa(input: str) -> str:
    """단일 후보 공약 질의 처리"""
    print("#### 단일 후보 공약 질의 Tool 활성화....")
    print("#### 후보자 이름 추출 중....")
    
    # 후보자 이름 추출
    result = candidate_chain.invoke({"question": input})["text"]
    target = next((c for c in candidates if c in result), None)
    
    if not target:
        return "Final Answer: 후보 이름을 인식할 수 없습니다."

    # 공약 검색
    print(f"#### {target} 후보자 공약집 검색 중....")
    result_dict = qa_chains[target].invoke({"question": input})
    answer = result_dict["answer"].strip()
    sources = result_dict.get("source_documents", [])

    # 관련성 검증
    print("#### 관련성 체크 중....")
    if is_empty_or_irrelevant(answer) or is_llm_irrelevant(input, answer) or len(sources) <= 1:
        return "Final Answer: 공약 없음"
    
    # 후처리
    print("#### 단일 후보 공약 RAG 실행 완료 ####")
    print("#### 사후 포맷팅 진행 ####")
    print("#"*50)
    
    answer = translate_if_needed(answer)
    final_text = single_candidate_formatter_chain.invoke({
        "question": input,
        "summaries": answer,
        "sources": sources,
    })["text"]

    global last_observation_output
    last_observation_output = final_text
    
    return f"Final Answer:\n{final_text}"

def run_policy_compare_all(input: str) -> str:
    """다자 후보 공약 비교"""
    print("\n#### 후보별 공약 비교 Tool 활성화....")
    print("##### 후보자 이름 추출 중....")
    
    # 입력 파싱
    if "," in input:
        split = [c.strip() for c in input.split(",")]
        involved = [c for c in split if c in candidates]
        keyword = next((k for k in split if k not in candidates), input)
    else:
        involved = candidates
        keyword = input.strip()

    print(f"#### 후보자 목록: {involved}")
    
    # 후보별 공약 수집
    comparisons = []
    for cand in involved:
        print(f"#### {cand} 후보자 공약집 검색 중....")
        result_dict = qa_chains[cand].invoke({"question": keyword})
        answer = result_dict["answer"]
        sources = result_dict.get("source_documents", [])
        
        if is_empty_or_irrelevant(answer) or is_llm_irrelevant(input, answer) or len(sources) <= 1:
            answer = "공약 없음"
        
        answer = translate_if_needed(answer)
        comparisons.append(format_candidate_policy(cand, answer))

    # 비교 분석
    print(f"#### 공약 비교 중....")
    result = compare_chain.invoke({
        "topic": keyword,
        "comparisons": "\n\n".join(comparisons)
    })["text"]
    
    result = translate_if_needed(result)
    print("#### 후보별 공약 비교 완료 ####")
    print("#"*50)

    # 최종 출력 구성
    final_text = f"{format_final_comparison(keyword, comparisons)}\n\n✅ 최종 비교 분석\n{result}"
    
    global last_observation_output
    last_observation_output = final_text
    
    return f"Final Answer:\n{final_text}"

def run_user_profile_recommendation(question: str) -> str:
    """사용자 프로파일 기반 추천"""
    print("\n#### 사용자 컨텍스트 별 질의 응답 Tool 활성화....")
    print("##### 사용자 프로파일 추출 중....")
    
    # 프로파일 추출
    profile_result = extract_user_profile_chain.invoke({"question": question})["text"]
    
    # 키워드 추출
    keywords = re.findall(r'관심정책 키워드\s*\(.*?\):\s*(.*)', profile_result)
    keyword_list = keywords[0].split(",") if keywords else []
    
    # 후보별 요약
    print("##### 후보별 공약 요약 중....")
    summaries, comparison_rows = summarize_all_candidates(
        [kw.strip() for kw in keyword_list], 
        question
    )
    
    # 추천 생성
    recommendation = recommend_chain.invoke({
        "question": question,
        "profile": profile_result,
        "summaries": summaries
    })["text"]
    
    # 출력 포맷팅
    final_text = format_recommendation_output(
        translate_if_needed(question),
        comparison_rows,
        recommendation
    )
    
    print("#### 사용자 컨텍스트 별 질의 응답 완료 ####")
    print("#"*50)
    
    global last_observation_output
    last_observation_output = final_text
    
    return "Final Answer:\n" + final_text

def run_policy_news_issue(input: str) -> str:
    """실시간 뉴스 이슈 분석"""
    print("\n#### 실시간 여론 반응 분석 Tool 활성화....")
    print("#### 후보자 이름 추출 중....")
    
    # 후보자 추출
    result = candidate_chain.invoke({"question": input})["text"]
    candidate = next((c for c in candidates if c in result), None)
    
    if not candidate:
        return "Final Answer: 후보 이름을 인식할 수 없습니다."
    
    # 키워드 추출
    keyword = input.replace(candidate, "").strip(" ,")
    if not keyword:
        return "Final Answer: 정책 주제(키워드)를 포함해 질문해 주세요."
    
    # 뉴스 분석
    print(f'#### {candidate} 후보의 {keyword} 관련 정책 뉴스 검색 중...')
    try:
        news_summary = news_final_summary(candidate, keyword)
        
        final_text = news_issue_summary_chain.invoke({
            "candidate": candidate,
            "keyword": keyword,
            "news_summary": news_summary
        })["text"]
        
        print("#### 실시간 여론 반응 분석 완료 ####")
        print("#"*50)
        
        global last_observation_output
        last_observation_output = final_text
        
        return final_text or f"{candidate} 후보의 '{keyword}' 관련 정책 정보를 외부에서 찾을 수 없습니다."
    except Exception as e:
        return f"{candidate} 후보의 '{keyword}' 관련 외부 뉴스 요약 중 오류가 발생했습니다: {e}"

# 보조 함수
def summarize_all_candidates(keywords: List[str], question: str) -> tuple[str, List[Dict]]:
    """후보별 공약 요약 수집"""
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
            
            extract = detailed_policy_chain.invoke({
                "question": question, 
                "summary": joined_summary
            })["text"]
            
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
            comparison_rows.append({
                "후보": cand, 
                "핵심 공약": "요약 실패", 
                "실현 방식": "", 
                "강점": ""
            })
    
    return "\n\n".join(summaries), comparison_rows

def build_bullet_summary(rows: List[Dict]) -> str:
    """불릿 포맷 요약 생성"""
    lines = ["[후보별 공약 요약]"]
    for row in rows:
        lines.append(f"- {row['후보']} 후보:")
        lines.append(f"  • 핵심 공약: {row['핵심 공약']}")
        lines.append(f"  • 실현 방식: {row['실현 방식']}")
        lines.append(f"  • 강점: {row['강점']}")
    return "\n".join(lines)

def format_recommendation_output(question: str, comparison_rows: List[Dict], recommendation_text: str) -> str:
    """추천 출력 포맷팅"""
    bullets = build_bullet_summary(comparison_rows)
    return f"""[질문]
{question}

{bullets}

[추천 요약]
{recommendation_text}"""

def format_final_comparison(topic: str, comparisons: List[str]) -> str:
    """최종 비교 포맷팅"""
    return f"""✅ '{topic}'에 대한 후보별 공약 비교 분석 결과\n\n{chr(10).join(comparisons)}"""
