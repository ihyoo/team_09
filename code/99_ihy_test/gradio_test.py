import gradio as gr

def answer_question(question):
    # 실제 로직은 여기에 넣으면 됩니다
    return f"📌 질문에 대한 응답: {question[::-1]}"  # 단순 예시 (역순 반환)

def reset():
    return "", ""  # 입력, 출력 모두 초기화

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🎯 질문-응답 데모")
    gr.Markdown("질문을 입력하고 '질문하기'를 누르세요. 초기화 버튼으로 모두 비울 수 있습니다.")

    with gr.Row():
        question_input = gr.Textbox(label="질문 입력", placeholder="예: AI는 무엇인가요?", lines=2)
        ask_button = gr.Button("❓ 질문하기", variant="primary")
        reset_button = gr.Button("🔄 초기화", variant="secondary")

    result_output = gr.Textbox(label="📥 응답 결과", lines=4, interactive=False)

    # 버튼 동작 설정
    ask_button.click(fn=answer_question, inputs=question_input, outputs=result_output)
    reset_button.click(fn=reset, outputs=[question_input, result_output])

demo.launch()