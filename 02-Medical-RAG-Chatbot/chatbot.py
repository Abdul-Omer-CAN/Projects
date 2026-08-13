## Imports ##

from langchain.document_loaders import TextLoader # Loads textfiles into LangChain. Reads our medical knowledge base text file.
from langchain.text_splitter import RecursiveCharacterTextSplitter # Splits large documents into smaller chunks. Splits  approx 500 char chunks to LLM.
from langchain.vectorstores import FAISS # Facebook AI Simliarity Search. Stores all text chunks as vectors. When questions comes in finds most simliar chunk.
from langchain.embeddings import HuggingFaceEmbeddings # Converts text to vectors.
from langchain.chains import RetrievalQA # The complete RAG pipeline in one object. Retrieval finds relevant chunks from FAISS. QA is Question Answering -> generates answers using LLM.
from langchain.llms import HuggingFacePipeline # Wraps a HuggingFace model so LangChain can use it. Basically converts HF language to Langchain language, so that it can understand.
from transformers import pipeline # Creates a text generation pipeline. Loads the actual LLM that generates the answer.
import torch # Pytorch backend. Powers the neural network under the hood.

## Load Knowledge Base ##

loader = TextLoader('medical_knowledge.txt') # Load our medical text file
documents= loader.load() # convert to LangChain documents
print(f"Loaded {len(documents)} documents") # counts how many documents were loaded(len).

## Split documents into Chunks ##

text_splitter = RecursiveCharacterTextSplitter( # creates a text_splitter object on how to split docs.
    chunk_size=500, # each chunk = 500 char
    chunk_overlap=50 # 50 char overlap between chunks. e.g last 50 char of chunk = first 50 of chunk 2 etc. With overlap if paragraph gets cutoff then it can lose context.
)

chunks = text_splitter.split_documents(documents) # split into chunks. returns a list of smaller document pieces. Each chunk = searchable unit in FAISS.
print(f"Split into {len(chunks)} chunks") # shows how many chunks were created.

## Create Vector Store ##

embeddings = HuggingFaceEmbeddings( # creates an embedding model. This model converts text -> vectors
    model_name="sentence-transformers/all-MiniLM-L6-v2" # small fast embedding model. Minilanguage model = MiniLM | L6 = neural network is 6 layers deep. More layers means more powerful but will be slower. | v2 = second version of the model.
)

vectorstore = FAISS.from_documents(chunks, embeddings) # Takes our text chunks & converts each chunk to vectors. Stores all vectors in FAISS database. FAISS can now search for simliar vectors instantly.
print("Vector store created!") # confirms everything worked.

## Load LLM ##

pipe = pipeline(
    "text2text-generation",  # Reads the question and answers appropriately. It is a text generation model.
    model="google/flan-t5-small" # small fast model. Better than GPT 2 for answering.
    max_new_tokens=200,  # max response length.
    torch_dtype=torch.float32 # use float32 for CPU. float16 causes error on cpu. float32 is more stable and safe. Tells pytorch what number format to use for the model's weights.
)

llm = HuggingFacePipeline(pipeline=pipe) # wrap for LangChain
print("LLM Loaded!")