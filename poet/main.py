# from dotenv import load_dotenv
# load_dotenv()


# from langchain.llms import OpenAI
# # 글을 이어서 작성해줌
# llm = OpenAI()
# result = llm.invoke('hi?')
# print(result)


# 언어 모델 예제
# from langchain_openai import ChatOpenAI
# # 글에 대답해줌
# chat_model = ChatOpenAI()
# result = chat_model.invoke('hi?')
# print(result)


# chat 모델 예제
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(openai_api_key = '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')
# llm.invoke("how can langsmith help with testing?")


# 인공지능 시인 만들기
import time
import streamlit as st
from langchain_openai import ChatOpenAI
import os
api_key = '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'
os.environ['OPENAI_API_KEY'] = api_key

chat_model = ChatOpenAI()

st.title('인공지능 시인')
content = st.text_input('시의 주제를 제시해주세요.') # 시의 주제 입력 받기

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중...'):
        time.sleep(5)
        st.write('시의 주제는', content, '입니다.')
        result = chat_model.invoke(content + "에 대한 시를 써줘") # 시 생성