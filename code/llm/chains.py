"""
make chains
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from data.document_loader import retrievers
from data.candidates import candidates
from llm.prompts import (
    candidate_detect_prompt,
    multi_candidate_comparison_prompt,
    user_profile_extraction_prompt,
    candidate_recommendation_prompt,
    relevance_prompt,
    translation_prompt,
    policy_element_extraction_prompt,
    news_issue_summary_prompt,
    fomatting_single_candidate_prompt,
    single_candidate_policy_prompt
)

# 1. LLM 모델 초기화
llm = ChatOpenAI(temperature=0.2)

# 2. 후보별 QA 체인 설정
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

# 3. 기능별 체인 초기화
candidate_chain = LLMChain(llm=llm, prompt=candidate_detect_prompt)
compare_chain = LLMChain(llm=llm, prompt=multi_candidate_comparison_prompt)
extract_user_profile_chain = LLMChain(llm=llm, prompt=user_profile_extraction_prompt)
recommend_chain = LLMChain(llm=llm, prompt=candidate_recommendation_prompt)
relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
detailed_policy_chain = LLMChain(llm=llm, prompt=policy_element_extraction_prompt)
news_issue_summary_chain = LLMChain(llm=llm, prompt=news_issue_summary_prompt)
single_candidate_formatter_chain = LLMChain(llm=llm, prompt=fomatting_single_candidate_prompt)

# 4. 에이전트 툴 설정
react_tools = [
    Tool(name="CandidatePolicyQA", func=run_candidate_policy_qa, 
         description="후보 이름과 주제를 기반으로 공약을 PDF에서 찾고, 공약 없음이면 바로 종료한다."),
    Tool(name="ComparePolicies", func=run_policy_compare_all, 
         description="복수 후보 간 특정 주제에 대한 공약을 비교합니다."),
    Tool(name="RecommendCandidateNatural", func=run_user_profile_recommendation, 
         description="사용자 질문에서 상황을 추출하고 적합한 후보를 추천합니다."),
    Tool(name="PolicyNewsIssueCheck", func=run_policy_news_issue,
         description="후보와 정책 키워드에 대해 최근 뉴스나 논란, 실제 이행 상황 등 사회적 이슈를 요약합니다.")
]

# 5. 에이전트 초기화
agent = initialize_agent(
    tools=react_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={
        "system_message": (
            "당신은 반드시 한국어로 사고하고 응답하는 정책 분석 도우미입니다. "
            "Thought, Action, Observation, Final Answer 형식을 사용하며, "
            "**Final Answer로 시작하는 응답이 나오면 체인을 반드시 종료하십시오.**"
        )
    },
    handle_parsing_errors=True
)

# 6. 쿼리 실행 함수
def action_query(query: str) -> str:
    response = agent.invoke({"input": query})["output"]
    
    if is_english(response):
        translated = translation_chain.invoke({"english_text": response})
        return translated["text"].strip() if isinstance(translated, dict) else translated
    return response
