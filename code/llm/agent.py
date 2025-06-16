from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from llm.chains import llm, react_tools

# 에이전트 초기화
agent = initialize_agent(
    tools=react_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={
        "system_message": (
            "당신은 반드시 한국어로 사고하고 응답하는 정책 분석 도우미입니다.\n"
            "Thought, Action, Observation, Final Answer 형식을 사용하며,\n"
            "**Final Answer로 시작하는 응답이 나오면 체인을 반드시 종료하십시오.**"
        )
    },
    handle_parsing_errors=True
)

def action_query(query: str) -> str:
    """사용자 질의 처리 메인 함수"""
    response = agent.invoke({"input": query})["output"]
    
    # 영어 응답 번역 처리
    if is_english(response):
        translated = translation_chain.invoke({"english_text": response})
        return translated["text"].strip() if isinstance(translated, dict) else translated
    return response

# 유틸리티 함수 임포트
# 순환 참조(circular import) 방지나 의존성 문제
from utils.helpers import is_english, translate_if_needed
from llm.chains import translation_chain
