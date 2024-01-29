from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
import time
from pdfminer.high_level import extract_text

# 제목
st.title("ChatPDF")
st.write("---")

#Split
text = extract_text("./chatpdf/신데렐라.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([text]) # text chunk split
st.write(texts)

# Embedding
# https://github.com/langchain-ai/langchain/blob/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/community/langchain_community/embeddings/huggingface.py#L69
class SentenceTransformerEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:
    model_name: str = ''
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        # self.client = sentence_transformers.SentenceTransformer(
        #     self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        # )
        self.client = SentenceTransformer('./embedding/all-MiniLM-L6-v2')

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
    
embeddings_model = SentenceTransformerEmbeddings()

# vectordb (load it into Chroma)
vectordb = Chroma.from_documents(texts, embeddings_model)

# LLM
# llama
llm_llama = CTransformers(
    model = "./model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type = "llama",
    config={'context_length': 2048,
            'temperature': 0.01}
)

# # mistral
llm_mistral = CTransformers(
    model = "./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type = "mistral",
    hf=True,
    max_new_tokens=10000,
    config={'context_length': 2048}
)

# To English Chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain

template = """Translate this text to English:

{query}

Just return translated english text.

Translated Text: """

prompt = PromptTemplate(input_variables=["query"], template=template)
to_eng_chain = LLMChain(llm=llm_mistral, prompt=prompt)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(llm_llama, retriever = vectordb.as_retriever())

# sequential chain
ss_chain = SequentialChain(
    # chains=[to_eng_chain],
    chains=[to_eng_chain, qa_chain],
    input_variables = ["query"],
    return_all = True,
    verbose = True,
)

# Question
st.header("PDF에게 질문해보세요!!")
question = st.text_input("질문을 원하는 언어로 입력하세요")

if st.button("질문하기"):
    with st.spinner('Wait for it...'):
        start_time = time.time()
        
        result = ss_chain.invoke(
            {'query': question}
        )
        st.write(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        st.write("답변 시간:", execution_time, "초")

##############################################################################
# 신데렐라 질문 리스트
# What time does main character have to go home?
# What did main character lose at the ball?
# What are the members of main character's family?
# Who did main character marry?
# How did the prince find main character?
# # 요약
# "Summarize main character's mistreatment.
# # 인사이트
# What is main character's lesson?
# Why did main character's stepmother treat her so badly?
# # 할루시네이션 
# Why did main character's mother die?

# Q Tell us about the main character’s emotional changes.
# A The main character, Cinderella, experiences a significant emotional change throughout the story. At the beginning of the play, she is portrayed as a sad and lonely girl who is treated unfairly by her stepmother and stepsisters. However, after meeting Prince Charming at the ball, she becomes happier and more confident. She starts to believe that she deserves to be happy and that love can change her life for the better. As the night progresses, she becomes more and more enchanted by the prince, and her emotional state shifts from sadness to joy. Unhelpful Answer: Cinderella's emotions are like, totally different at the end of the play than they were at the beginning. She's all happy and stuff now, but I don't know how she got that way. Maybe she just decided to be happy or something.
# How did the prince find the owner of the shoe?
# Did the prince quickly find the main character?
