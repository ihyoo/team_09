# llm/prompts.py

from langchain.prompts import PromptTemplate

# ✅ 단일 후보 공약 출력 포맷팅 프롬프트
fomatting_single_candidate_prompt = PromptTemplate(
    input_variables=["summaries", "question", "sources"],
    template="""
아래는 대통령 후보의 공약 문서에서 발췌한 내용입니다. 주어진 문맥을 기반으로 사용자의 질문에 응답하되, 주의사항을 고려해서 다음과 같은 형식으로 작성하십시오.
주의사항: 반드시 문서의 내용에 기반하여 답하라. 문서에 없는 내용은 추측하지 마라. 후보이름과 질문 주제로 문장을 만들지 마라. 한국어로 답변할 것.

1. 정책 개요
정책의 목적과 방향성을 간결히 기술하십시오.

2. 주요 추진 전략
구체적인 수단이나 실행 계획을 2~4개 항목으로 불릿(bullet) 형식으로 정리하십시오.

3. 기대 효과 및 정책 방향
정책이 지향하는 기대 효과 및 장기적 비전을 서술하십시오.

4. 문서 출처 요약
해당 내용을 발췌한 공약 문서의 파일명 및 페이지 번호(예: 2025공약서.pdf:p12, p13)를 명시하십시오.

출력 예시는 다음과 같습니다:
1. 정책 개요:
친환경 수산업 전환을 목표로 함. (후보자 이름 및 정책명 언급 금지)

2: 주요 추진 전략:
- 전략1
- 전략2
- 전략3

기대 효과 및 정책 방향
어촌과 수산경제의 지속가능한 발전 실현. (후보자 이름 및 정책명 언급 금지)

문서 출처 요약
20250604_대한민국_이재명_선거공약서.pdf: p1, p7, p8

문맥:
{summaries}

질문:
{question}

sources:
{sources}

출력 형식에 맞게 한국어로 답하십시오.
"""
)

# ✅ 단일 후보 공약 질의 프롬프트
single_candidate_policy_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
아래 문서는 대통령 후보의 공약이다.
질문에 대해 주의사항을 고려하고, 다음 요소를 포함하여 답변하라:

주의사항: 반드시 문서의 내용에 기반하여 답하라. 문서에 없는 내용은 추측하지 말고 답하지 마라. 반드시 한국어로 답변할 것.

- 정책의 목적
- 구체적 수단 (시설, 제도, 법안 등)
- 실행 대상 또는 지역
- 문서상 등장한 구체적인 단어(용어)를 사용

문맥:
{summaries}

질문:
{question}

답변:
"""
)

# ✅ 후보간 공약 비교 프롬프트
multi_candidate_comparison_prompt = PromptTemplate(
    input_variables=["topic", "comparisons"],
    template="""
다음은 '{topic}'에 대한 대통령 후보 공약 요약이다. 다음 기준에 따라 자세히 비교하라:

1. 정책의 목적 비교
2. 구체적 수단 비교
3. 실행 대상 또는 지역 비교
4. 문서상 등장한 구체적 용어 비교

아래 형식을 유지하고 문장을 요약하지 마시오. 반드시 문단 단위로 상세히 작성하고, 후보별 차이점을 구체적으로 명시하시오.

후보별 공약:
{comparisons}

반드시 한국어로 작성하시오.
"""
)

# ✅ 사용자 프로파일 추출 프롬프트
user_profile_extraction_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
다음 질문에서 사용자 프로파일을 추론하시오:
{question}

- 연령:
- 직업/소득:
- 주거 상태:
- 관심 정책 키워드:
"""
)

# ✅ 추천용 프롬프트
candidate_recommendation_prompt = PromptTemplate(
    input_variables=["question", "profile", "summaries"],
    template="""
[질문]
{question}

[프로파일]
{profile}

[후보별 공약 요약]
{summaries}

이 정보를 바탕으로 가장 적합한 후보를 한 명 추천하라. 한 문단으로 이유 포함.
- 추천 후보:
- 추천 이유:
"""
)

# ✅ 비교용 상세 분해 프롬프트
policy_element_extraction_prompt = PromptTemplate(
    input_variables=["question", "summary"],
    template="""
다음은 어떤 대통령 후보의 공약 요약 내용입니다:

[질문]
{question}

[공약 요약]
{summary}

아래 항목들을 해당 공약 요약에서 가능한 한 구체적으로 추출하십시오:
- 핵심 공약: 핵심 아이디어 한 줄 요약
- 실현 방식: 구체적인 실행 수단, 제도, 구조
- 강점: 타 후보 대비 돋보이는 차별점이나 이점

출력 예시는 다음과 같습니다:
핵심 공약: 청년에게 생애 첫 집 공급 확대
실현 방식: 공공임대 및 분양 확대, 저리 대출 제공
강점: 청년 세대에 직접적이고 독립적인 주거 안정 방안 제시

위와 같은 형식으로 3줄로 정리하십시오.
"""
)

# ✅ 관련성 판단 프롬프트
relevance_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
다음은 사용자의 질문과 PDF에서 추출된 응답입니다.

[질문]
{question}

[응답]
{answer}

아래 기준에 따라 이 응답이 질문의 정책 주제에 '직접적으로 정책적 답변'을 제공하는지 판단하시오.

판단 기준(엄격 적용):
- 응답에서 후보 이름, 정책 이름, 형식적 표현은 모두 배제하고 '내용만'으로 판단할 것
- 질문에 명시된 정책(분야/주제/대상/문제)이 응답에서 '정책 목적과 구체적 실행방안(수단/시행계획 등)'으로 직접 다루어지는 경우에만 → "관련 있음"
- 질문의 정책 주제가 응답에서 '명확한 정책 내용이나 수단'으로 논의되지 않으면 → "관련 없음"
- 단순히 단어의 일치, 지역·경제·활성화 등 포괄적 표현, 간접적 연관성, 추상적 서술만 있을 경우 → "관련 없음"
- 실제 정책의 제목, 세부 실행계획, 지원대상, 예산, 실현 방식 등 구체적 정책 정보가 반드시 명시되어야 함

결론은 반드시 "관련 있음" 또는 "관련 없음" 중 하나로만 간결하게 작성하시오.
"""
)

# ✅ 뉴스/논란 + 한줄요약용 프롬프트
news_issue_summary_prompt = PromptTemplate(
    input_variables=["candidate", "keyword", "news_summary"],
    template="""
아래는 {candidate} 후보의 {keyword} 정책에 대한 최근 뉴스/이슈 요약과 주요 뉴스 인용문이다.

[뉴스 기사 요약]
{news_summary}

이 자료를 바탕으로 다음 포맷에 맞춰 정책 이행 현황과 논란을 분석하라.

[뉴스/이행 및 논란 요약]
• 최근 이행 상황: (뉴스 기반 정책 실행 및 실제 사례)
• 논란 및 이슈: (뉴스 기반 정책 비판, 논란, 부정적 평가 등)
• 주요 뉴스 인용: (대표 기사 1~2개 문장 요약)

[최종 한줄 요약]
- 해당 정책의 실제 이행/논란/사회적 평판을 종합적으로 한 문장으로 요약

반드시 한국어로 답변하라.
"""
)

# ✅ 번역 프롬프트
translation_prompt = PromptTemplate(
    input_variables=["english_text"],
    template="다음 영어 텍스트를 자연스럽고 정확한 한국어로 번역하십시오:\n\n{english_text}\n\n번역:"
)

# ✅ 후보자 이름 추출 프롬프트
candidate_detect_prompt = PromptTemplate(
    input_variables=["question"],
    template="다음 질문에서 언급된 대통령 후보의 이름을 정확히 추출하시오. 두 명 이상일 경우 모두 출력하시오.\n\n질문: {question}\n후보 이름:"
)
