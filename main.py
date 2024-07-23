import os
import warnings
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Azure Chat OpenAI model
llm = AzureChatOpenAI(
    azure_deployment="v1",
    temperature=0
)

# Replace with your preferred SentenceTransformer model
model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

# Custom embedding class
class CustomEmbeddings(Embeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, texts):
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding.tolist()

embeddings = CustomEmbeddings(embedding_model)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define request and response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Global variable for the rag_chain
rag_chain = None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_chain
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    # Save the uploaded file to a temporary location
    file_location = f"temp/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())

    # Load and process the PDF file
    loader = PyPDFLoader(file_location)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create the vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Retrieve and generate using the relevant snippets of the PDF
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return {"detail": "PDF uploaded and processed successfully"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet")

    user_input = request.message

    if not user_input:
        raise HTTPException(status_code=400, detail="No message provided")

    # Generate a response using the RAG chain
    response = rag_chain.invoke(user_input)

    return ChatResponse(response=response)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
