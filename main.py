import os
import shutil
from typing import List, Optional, AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import SecretStr, BaseModel

load_dotenv()

# --- FastAPI Setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Global Variables & Configuration ---
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

PDFS_DIR = "./data/pdfs"
os.makedirs(PDFS_DIR, exist_ok=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=SecretStr(gemini_key))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(gemini_key))

# Global retriever instance, initialized to None
retriever = None
qa_chain: Optional[RunnableSerializable] = None

# --- Helper Functions ---
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs

def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain():
    """Initializes the RAG chain with a retriever."""
    global qa_chain
    if retriever is None:
        print("Retriever not initialized. Cannot create RAG chain.")
        return

    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.

Context: {context}
Question: {question}
Helpful Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain initialized successfully.")


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF upload, processing, and vector store creation."""
    global retriever
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # Clean up PDF directory only
    if os.path.exists(PDFS_DIR):
        shutil.rmtree(PDFS_DIR)
    os.makedirs(PDFS_DIR)

    file_path = os.path.join(PDFS_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=400, detail="Could not load any documents from the PDF.")

        # 2. Filter and Split
        docs = filter_to_minimal_docs(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        # 3. Create In-Memory Vector Store - NO file persistence
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings
            # No persist_directory - this creates an in-memory store
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        # 4. Initialize the RAG chain
        initialize_rag_chain()

        return JSONResponse(
            content={"message": "PDF processed successfully!", "filename": file.filename},
            status_code=200
        )
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {e}")
    finally:
        # Close the file to ensure it's handled properly by the OS
        file.file.close()


class ChatRequest(BaseModel):
    message: str

async def get_streamed_response(message: str) -> AsyncGenerator[str, None]:
    """Wraps the RAG chain's stream in an async generator, formatting for SSE."""
    if not qa_chain:
        # This case should ideally be handled before calling, but as a safeguard:
        return

    async for chunk in qa_chain.astream(message):
        # Format as a Server-Sent Event
        yield f"data: {chunk}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles chat messages by querying the RAG chain and streaming the response."""
    if not qa_chain:
        raise HTTPException(status_code=400, detail="RAG chain not initialized. Please upload a PDF first.")

    try:
        return StreamingResponse(get_streamed_response(request.message), media_type="text/event-stream")
    except Exception as e:
        print(f"Error during chat invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

# To run this app: uvicorn main:app --reload
