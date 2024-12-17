import torch
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain import WikipediaAPIWrapper
# from langchain.llms import HuggingFacePipeline
from langchain.embeddings.base import Embeddings
import transformers
import numpy as np


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str | int):
        self.pipeline = transformers.pipeline("feature-extraction", model=model_name, device=device)

    def embed_documents(self, texts):
        embeddings = self.pipeline(texts)
        processed_embeddings = []
        for embedding in embeddings:
            flattened = np.array(embedding).flatten()

            # Choose a fixed embedding size
            target_size = 384

            if flattened.size > target_size:
                processed_embedding = flattened[:target_size]
            else:
                processed_embedding = np.pad(
                    flattened,
                    (0, max(0, target_size - flattened.size)),
                    mode='constant')
            processed_embeddings.append(processed_embedding)

        return processed_embeddings

    def embed_query(self, text):
        embedding = self.pipeline(text)
        flattened = np.array(embedding[0]).flatten()
        target_size = 384

        if flattened.size > target_size:
            processed_embedding = flattened[:target_size]
        else:
            processed_embedding = np.pad(
                flattened,
                (0, max(0, target_size - flattened.size)),
                mode='constant'
            )

        return processed_embedding


class WikipediaQAAgent:
    safety_alert_message = "Sorry, I can't help with that."
    # TODO: might change this message to a better one?

    def __init__(self, device):
        self.embedding_model = HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2", device=device)
        # TODO: make embeddings models loaded from locally saved weights

        self.answer_llm = transformers.pipeline("text-generation", model="NousResearch/Hermes-3-Llama-3.2-3B",
                                                device=device, max_length=1000, truncation=True)
        # TODO: change answer_model to a locally saved model; also change the model to llama 3.1 7b-instruct
        # I have already downloaded llama 3.1 7b-instruct on my google drive.

        self.wikipedia_tool = WikipediaAPIWrapper()
        self.guardrail = None    # TODO: add guardrail

    def __call__(self, question: str) -> str:
        return self._qa_pipeline(question)

    def _retrieve_wikipedia_passages(self, question: str):
        raw_results = self.wikipedia_tool.run(question)
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        passages = text_splitter.split_text(raw_results)
        return [Document(page_content=passage) for passage in passages]

    def _create_vector_store(self, documents):
        print(documents)
        return FAISS.from_documents(documents, self.embedding_model)

    def _qa_pipeline(self, question: str) -> str:
        documents = self._retrieve_wikipedia_passages(question)
        vector_store = self._create_vector_store(documents)
        retriever = vector_store.as_retriever()
        relevant_passages = retriever.get_relevant_documents(question)
        combined_passages = "\n".join([doc.page_content for doc in relevant_passages])
        prompt = f"Question: {question}\nPassages:\n{combined_passages}\nAnswer:"
        is_safe = self.guardrail(prompt)
        if not is_safe:
            return WikipediaQAAgent.safety_alert_message
        final_answer = self.answer_llm(prompt)
        return final_answer

    def _batch_qa_pipeline(self, questions: list[str]) -> list[str]:
        # TODO: complete this
        pass
