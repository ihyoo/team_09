from langchain.agents import Tool, initialize_agent, AgentType
from chains.chains import *
from core.utils import is_english

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

def action_query(query: str) -> str:
    response = agent.invoke({"input": query})["output"]
    if is_english(response):
        translated = translation_chain.invoke({"english_text": response})
        return translated["text"].strip() if isinstance(translated, dict) else translated
    return response
