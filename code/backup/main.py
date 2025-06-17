
import os
import openai
from dotenv import load_dotenv
import argparse
import gradio as gr

from llm.agent import action_query

def cli_main():
    """기존 CLI 인터페이스"""
    print("🇰🇷 정책 분석 도우미 서비스 시작")
    print("'exit' 입력 시 종료\n" + "="*50)
    
    while True:
        try:
            query = input("\n질문을 입력하세요: ")
            if query.lower() in ['exit', 'quit']:
                print("서비스를 종료합니다.")
                break
                
            result = action_query(query)
            print("\n" + "="*50)
            print(result)
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n사용자 요청으로 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {str(e)}")

def gradio_interface():
    """Gradio 웹 인터페이스"""
    def answer_question(question):
        try:
            return action_query(question)
        except Exception as e:
            return f"❌ 오류 발생: {str(e)}"

    def reset():
        return "", ""

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("## 🎯 정책 분석 도우미")
        gr.Markdown("### 후보별 공약 분석 및 비교 서비스")

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="질문 입력",
                    placeholder="예: 이재명 후보의 주거 정책과 김문수 후보의 경제 정책을 비교해주세요",
                    lines=3
                )
                with gr.Row():
                    ask_btn = gr.Button("🔍 분석 실행", variant="primary")
                    reset_btn = gr.Button("🔄 초기화", variant="secondary")

            with gr.Column(scale=2):
                result_output = gr.Textbox(
                    label="📄 분석 결과",
                    lines=8,
                    interactive=False
                )

        ask_btn.click(fn=answer_question, inputs=question_input, outputs=result_output)
        reset_btn.click(fn=reset, outputs=[question_input, result_output])

    demo.launch(server_port=7860, share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradio", action="store_true", help="Gradio 웹 인터페이스 실행")
    args = parser.parse_args()

    # os.environ['OPENAI_API_KEY'] = input("Enter your OpenAI API key: ")
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if args.gradio:
        gradio_interface()
    else:
        cli_main()
