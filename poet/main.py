# from dotenv import load_dotenv
# load_dotenv()

#############################################################################################################

# from langchain.llms import OpenAI
# # 글을 이어서 작성해줌
# llm = OpenAI()
# result = llm.invoke('hi?')
# print(result)

#############################################################################################################

# 언어 모델 예제
# from langchain_openai import ChatOpenAI
# # 글에 대답해줌
# chat_model = ChatOpenAI()
# result = chat_model.invoke('hi?')
# print(result)

#############################################################################################################

# chat 모델 예제
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(openai_api_key = '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')
# llm.invoke("how can langsmith help with testing?")

#############################################################################################################
# 인공지능 시인 만들기
# llama2 7b 경량화 모델
import time
import streamlit as st
from langchain_community.llms import CTransformers

llm = CTransformers(
    model = "./model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type = "llama"
)

# mistral 모델
# from ctransformers import AutoModelForCausalLM
# from transformers import pipeline, AutoModel, AutoTokenizer

# device = "cpu"

# model = AutoModelForCausalLM.from_pretrained('./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf', 
#                                              model_type="mistral", 
#                                             #  gpu_layers=20, 
#                                              hf=True,
#                                              max_new_tokens=10000,
#                                              context_length=10000)

# tokenizer = AutoTokenizer.from_pretrained('./tokenizer/')

# generator = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     framework='pt',
#     max_new_tokens=500,
#     repetition_penalty=1.1
# )

st.title('인공지능 시인')
content = st.text_input('시의 주제를 영어로 제시해주세요.') # 시의 주제 입력 받기

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중...'):
        start_time = time.time()
        # response = llm.invoke(content +  "에 대한 시(poet)를 작성해 주세요")
        response = llm.invoke("Write a poem about " + content)
        st.write(response)
        end_time = time.time()
        execution_time = end_time - start_time
        st.write("\n시 작성 시간:", execution_time, "초")

#############################################################################################################