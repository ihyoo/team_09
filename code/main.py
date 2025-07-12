import os
import re
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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


def is_llm_irrelevant(question: str, answer: str) -> bool:
    try:
        result = relevance_chain.invoke({"question": question, "answer": answer})
        result_text = result["text"].strip() if isinstance(result, dict) else result.strip()
        return "관련 없음" in result_text
    except Exception as e:
        print(f"❌ 관련성 판단 실패: {e}")
        return False


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


def run_candidate_policy_qa(input: str) -> str:
    result = candidate_chain.invoke({"question": input})["text"]
    target = next((c for c in candidates if c in result), None)
    if not target:
        return "Final Answer: 후보 이름을 인식할 수 없습니다."

    result_dict = qa_chains[target].invoke({"question": input})
    answer = result_dict["answer"].strip()
    sources = result_dict.get("source_documents", [])

    if is_empty_or_irrelevant(answer) or is_llm_irrelevant(input, answer) or len(sources) <= 1:
        return "Final Answer: 공약 없음"

    answer = translate_if_needed(answer)
    final_text = single_candidate_formatter_chain.invoke({
        "question": input,
        "summaries": answer,
        "sources": sources,
    })["text"]

    return final_text


def action_query(query):
    response = agent.invoke({"input": query})["output"]
    return translate_if_needed(response) if is_english(response) else response


def action_query_wrapper(question, state):
    result = action_query(question)
    detailed = f"[상세 요약]\n{result}"
    return result, detailed, result


def reset():
    return "", "", ""

# Setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.2)

# Prompt & Chains
translation_prompt = PromptTemplate(input_variables=["english_text"], template="다음 영어 텍스트를 자연스럽고 정확한 한국어로 번역하십시오:\n\n{english_text}\n\n번역:")
candidate_detect_prompt = PromptTemplate(input_variables=["question"], template="다음 질문에서 언급된 대통령 후보의 이름을 정확히 추출하시오. 두 명 이상일 경우 모두 출력하시오.\n\n질문: {question}\n후보 이름:")

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

candidate_chain = LLMChain(llm=llm, prompt=candidate_detect_prompt)
relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
detailed_policy_chain = LLMChain(llm=llm, prompt=policy_element_extraction_prompt)
single_candidate_formatter_chain = LLMChain(llm=llm, prompt=fomatting_single_candidate_prompt)

# Tools & Agent
tool_qa = Tool(name="CandidatePolicyQA", func=run_candidate_policy_qa, description="후보 이름과 주제를 기반으로 공약을 PDF에서 찾고, 공약 없음이면 바로 종료한다.")
react_tools = [tool_qa]

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

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🧭 유권자 나침반 질의-응답")
    gr.Markdown("### 대통령 후보 공약 기반 AI Agent")
    gr.Markdown("사용 방법 : 질문을 입력하고 '질문하기'를 누르세요. 초기화 버튼으로 모두 비울 수 있습니다.")

    with gr.Row():
        question_input = gr.Textbox(label="질문 입력", placeholder="예: 이재명 후보의 조선해양 정책에 대해 알려줘.", lines=2)
        ask_button = gr.Button("❓ 질문하기", variant="primary")
        reset_button = gr.Button("🔄 초기화", variant="secondary")

    result_output = gr.Textbox(label="📥 한 줄 요약 및 결과", lines=4, interactive=False)
    state_output = gr.Textbox(label="🧠 상세 요약", lines=6, interactive=False)
    output_state = gr.State(value="")

    ask_button.click(fn=action_query_wrapper, inputs=[question_input, output_state], outputs=[result_output, state_output, output_state])
    reset_button.click(fn=reset, outputs=[question_input, result_output, state_output])

# 실행
demo.launch()
