import os
import time
import ast
import re
import openai
from apis.naver_news import NaverNewsAPI
from dotenv import load_dotenv

# 환경 변수 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
news = NaverNewsAPI()

def gpt_prompt_action(prompt: str, max_tokens: int):
    """OpenAI GPT-3.5를 이용한 프롬프트 실행"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def summary(text: str, keyword: str) -> str:
    """뉴스 기사 키워드 요약"""
    prompt = f"""
너는 뉴스 분석 전문가야. 다음은 뉴스 기사 전문이야.

[뉴스 기사 전문]
{text}

이 기사에서 "{keyword}"와 관련된 내용이 있다면, 관련된 내용을 요약해서 알려줘.
만약 관련 내용이 없다면 부가 설명 없이 "관련 없음" 이란 단어만 말해줘.
"""
    return gpt_prompt_action(prompt, 300)

def news_filter(news_list: list, search_word: str) -> str:
    """뉴스 감성 분석 필터링"""
    filter_ls = [x.replace('"', '') for x in news_list if '관련 없음' not in x]
    count = len(filter_ls)
    concat_text = '\n'.join(filter_ls)
    
    prompt = f"""
아래 문장들을 보고 {search_word} 주제 기준으로 긍정적인지 부정적인지 알려줘.

{concat_text}

답변은 부가 설명없이 아래 list안에 json 형식을 담아서 답변해줘.
요소인 json 형식은 아래와 같아. 총 {count}개 문장이니 리스트에 요소 확실히 개수 맞춰서 대답해줘.

{{
    'num' : '위에 나오는 문장의 순서',
    'sentiment' : '긍정 or 부정'
}}

답변은 꼭 리스트로 해줘
    """
    return gpt_prompt_action(prompt, 4000)

def news_final_summary(candidate: str, search_word: str) -> str:
    """후보별 정책 최종 요약"""
    try:
        result = news.search(keyword=f"{candidate} {search_word}")
    except Exception as e:
        return f"뉴스 검색 중 오류 발생: {e}"

    check_ls = []
    for item in result.get("items", []):
        try:
            result = summary(item['title'] + item['description'], search_word)
            check_ls.append(result)
        except Exception as e:
            check_ls.append(f"요약 실패: {e}")

    filter_ls = [x.replace('"', '') for x in check_ls if '관련 없음' not in x]
    concat_text = '\n'.join(filter_ls)
    
    prompt = f"""
너는 정책 분석 전문가야.

아래 문장들은 다양한 뉴스 기사에서 추출된 내용이야.
이 문장들을 보고 {candidate} 후보의 "{search_word}" 주제와 직접적으로 관련된 정책이 있는지 판단해.

만약 {candidate} 후보의 정책이 명확하게 드러난다면, **구체적인 정책 내용을 요약해서 한 문단으로 작성해줘.**

하지만 다음 중 하나라도 해당된다면, 부가 설명 없이 반드시 "관련 없음" 이라고만 말해:
- 뉴스 내용이 {candidate} 후보의 정책이 아닌 경우
- {search_word} 주제와 직접 관련이 없는 경우
- 정책의 내용이 불분명하거나 판단이 애매한 경우

문장 목록:
{concat_text}

답변 형식:
- 관련 있을 경우: {candidate} 후보의 정책 요약 한 문단
- 관련 없을 경우: 관련 없음
    """
    return gpt_prompt_action(prompt, 4000)

def news_sentiment_action(search_word: str) -> str:
    """뉴스 감성 분석 주 실행 함수"""
    result = news.search(keyword=search_word)
    check_ls = []
    
    for item in result.get("items", []):
        result = summary(item['title'] + item['description'], search_word)
        check_ls.append(result)

    final_result = news_filter(check_ls, search_word)
    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            data = ast.literal_eval(final_result)
            if isinstance(data, list):
                total = len(data)
                positive = sum(1 for item in data if item['sentiment'] == '긍정')
                negative = sum(1 for item in data if item['sentiment'] == '부정')
                
                positive_ratio = round(positive / total * 100, 2)
                negative_ratio = round(negative / total * 100, 2)
                
                final_result = (f"{search_word} - "
                               f"긍정({positive_ratio}%) / 부정({negative_ratio}%)")
                return final_result
        except (ValueError, SyntaxError) as e:
            retries += 1
            time.sleep(1)
    
    return "감성 분석 실패: 최대 재시도 횟수 초과"
