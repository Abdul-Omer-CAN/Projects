## IMPORTS ##

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch

## Create FASTAPI #

app=FastAPI()

## Question Model ##

class Question(BaseModel):
    question: str

## Load knowledge BASE + Build RAG Chain (runs at startup) ##

loader = TextLoader('medical_knowledge.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

pipe = pipeline("text-generation", model="gpt2", max_new_tokens=200, dtype=torch.float32, pad_token_id=50256)
llm = HuggingFacePipeline(pipeline=pipe)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

print("RAG BOT ready to serve!")

## Ask Endpoint #
@app.post("/ask")
def ask(q: Question):
    result = qa_chain.invoke({"query": q.question})
    return {"question": q.question, "answer": result['result']}

# Run the SERVER #
if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8600)