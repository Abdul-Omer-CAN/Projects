## Imports ##

from langchain_community.document_loaders import TextLoader # Loads textfiles into LangChain. Reads our medical knowledge base text file.
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splits large documents into smaller chunks. Splits  approx 500 char chunks to LLM.
from langchain_community.vectorstores import FAISS # Facebook AI Simliarity Search. Stores all text chunks as vectors. When questions comes in finds most simliar chunk.
from langchain_community.embeddings import HuggingFaceEmbeddings # Converts text to vectors.
from langchain_classic.chains import RetrievalQA # The complete RAG pipeline in one object. Retrieval finds relevant chunks from FAISS. QA is Question Answering -> generates answers using LLM.
from langchain_huggingface import HuggingFacePipeline # Wraps a HuggingFace model so LangChain can use it. Basically converts HF language to Langchain language, so that it can understand.
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
    "text-generation",  # Reads the question and answers appropriately. It is a text generation model.
    model="gpt2", # small fast model. Better than GPT 2 for answering.
    max_new_tokens=200,  # max response length.
    dtype=torch.float32, # use float32 for CPU. float16 causes error on cpu. float32 is more stable and safe. Tells pytorch what number format to use for the model's weights.
    pad_token_id=50256
)

llm = HuggingFacePipeline(pipeline=pipe) # wrap for LangChain
print("LLM Loaded!")

## Build RAG chain ##

retriever = vectorstore.as_retriever( # Converts FAISS vector into a retriever(search interface ontop of the vector storage).
    search_kwargs={"k":3} # When questions comes in retrieve top 3 most simliar chunks.
)

qa_chain = RetrievalQA.from_chain_type( #  Creates the complete RAG pipeline. Connects the retriever + LLM together. 
    # One object that handles everything. "RetrievalQA" is the class name. Retrieval finds the relevant docs and QA is questioning and answering. 
    # Together finds docs and answers questions using them.
    llm=llm, # Pass our FLAN-T5 model. This generates the final answer.
    chain_type = "stuff", # take all three retrieval prompts and stuff them into one prompt.
    retriever = retriever, # Pass our FAISS retriever. This finds relevant chunks for each question.
    return_source_documents=True # Returns which chunks were used to answer. Good for transparency. See where the answer came from.
)
# FULL RAG Flow # : Question → retriever finds top 3 chunks → 
# stuffed into prompt → llm generates answer → 
# returned with source documents
print("RAG chain ready!")


## Test the RAG chatbot ##

def ask_question(question): # defines a function that takes a question as a input. Reusable fxn - call it with any medical question.
    result = qa_chain.invoke({"query": question}) # Calls on the RAG Chain.".invoke" runs the chain. "query" passes the question in dictionary format. Returns answers with source documents.
    print(f"\nQuestion: {question}") # Prints the question
    print(f"Answer: {result['result']}") # Prints the generated answer. Gets the actual answer from the result dictionary. result contains the generated answer and the source documents.
    print("---") # A seperator line between questions.

# Test questions #

ask_question("What causes heart disease?")
ask_question("What are the symptoms of diabetes?")
ask_question("How is high blood pressure treated?")

## Gets converted to vector
## FAISS finds top 3 similar chunks.( FAISS stores both as vectors and words)
## Chunks stuffed into prompt
## FLAN-T5 generates answer
## Answer printed! 😄