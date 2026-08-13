# Medical RAG Chatbot 🤖

A Retrieval Augmented Generation (RAG) chatbot that answers medical questions using a custom knowledge base.

## What it does
- Loads a medical knowledge base (heart disease, diabetes, blood pressure, cholesterol, obesity)
- Splits documents into searchable chunks
- Converts chunks into vectors using sentence embeddings
- Stores vectors in a FAISS vector database
- Retrieves the most relevant chunks for any medical question
- Generates natural language answers using an LLM grounded in retrieved context

## Tech Stack
- Python
- LangChain
- FAISS (vector database)
- Hugging Face Transformers
- Sentence Transformers (embeddings)
- GPT-2 (text generation)
- PyTorch

## How it Works (RAG Pipeline)
1. Medical text is loaded and split into 500-character chunks
2. Each chunk is converted into a vector using `all-MiniLM-L6-v2`
3. Vectors are stored in FAISS for fast similarity search
4. User question is converted into a vector
5. FAISS retrieves the top 3 most relevant chunks
6. Chunks + question are passed to the LLM
7. LLM generates an answer grounded in the retrieved medical context

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the chatbot
python chatbot.py

## Example Question
"What causes heart disease?"

## Example Output
Retrieves relevant chunks about heart disease, cholesterol, and blood pressure, then generates a grounded answer using those retrieved chunks as context.

## Known Limitations
- Uses GPT-2-small for demonstration purposes, which can occasionally hallucinate or ramble
- Production version would use a larger instruction-tuned model for cleaner, more accurate answers

## Dataset
Custom medical knowledge base covering heart disease, diabetes, high blood pressure, cholesterol, and obesity

## Author
Abdulrehman Omer | github.com/Abdul-Omer-CAN