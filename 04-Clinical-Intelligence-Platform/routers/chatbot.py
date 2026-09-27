from fastapi import APIRouter
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from typing import Optional, List
import torch
import os


router = APIRouter()

# Question model

class Question(BaseModel):
    question: str

# Load knowledgebase + build rag chain(only runs once)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
knowledge_path = os.path.join(BASE_DIR, 'medical_knowledge.txt')

loader = TextLoader(knowledge_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(chunks, embeddings)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", dtype=torch.float32)

class FlanT5LLM(LLM):
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=200, min_new_tokens=45)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
llm = FlanT5LLM()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context to answer the question in detail, with a full explanatory paragraph covering causes, symptoms, and treatment where relevant. Only include information that directly answers the question — do not include unrelated topics from the context. Do not answer in just one short sentence.
{context}

Question: {question}
Detailed Answer:"""
    )


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
    )
print("RAG chatbot router ready!")

# Ask endpoint

@router.post("/ask")
def ask(q: Question):
    result = qa_chain.invoke({"query": q.question})
    return {"question": q.question, "answer": result['result']}