from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# 1. Loader
loader = PyPDFLoader("./chatpdf/unsu.pdf")

# 2. Split
pages = loader.load_and_split() # page split

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages) # text chunk split

# 3. Embedding
# https://github.com/langchain-ai/langchain/blob/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/community/langchain_community/embeddings/huggingface.py#L69
from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

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
    
embedding_function = SentenceTransformerEmbeddings()

# 4. vectordb (load embeddings into ChromaDB)
vectordb = Chroma.from_documents(texts, embedding_function)

# 5. LLM
llm = CTransformers(
    model = "./model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type = "llama"
)

# 6. Question
question = "아내가 먹고 싶어하는 음식은 무엇이야?"

# # 7. Searching docs in vectordb
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=vectordb.as_retriever(), llm=llm
# )

# docs = retriever_from_llm.get_relevant_documents(query=question)
# print(len(docs))
# print(docs)

# 8. langchain result

qa_chain = RetrievalQA.from_chain_type(llm, retriever = vectordb.as_retriever())
result = qa_chain({'query': question})

print('답변 결과', result)

