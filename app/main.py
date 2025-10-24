# File: app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import uuid
import logging
from dotenv import load_dotenv
from typing import List
from contextlib import asynccontextmanager

# Load environment variables from .env file FIRST
load_dotenv()

# Import your custom modules
from app.utils import pdf_loader
from app.core import db, rag
from app.models import schemas

# --- App Initialization ---

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- UPDATED LIFESPAN FUNCTION ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs *before* the app starts accepting requests
    logger.info("Application startup... initializing Database.")
    
    # 1. Initialize DB and capture the returned object
    vector_store = db.initialize_database()
    
    logger.info("Initializing RAG chain.")
    
    # 2. Pass the object to the RAG initializer
    rag.initialize_rag_chain(vector_store)
    
    yield
    # This code runs *after* the app shuts down
    logger.info("Application shutdown.")

# --- Initialize FastAPI app ---
app = FastAPI(
    title="PanScience RAG API",
    description="An API for document upload and retrieval-augmented generation.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Environment Variable Check ---
if not os.getenv("GOOGLE_API_KEY"):
    logger.warning("GOOGLE_API_KEY environment variable not found. /query endpoint will fail.")

# --- Temporary Upload Directory ---
UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def get_status():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "message": "RAG API is running."}


@app.post("/upload", 
          tags=["Documents"], 
          response_model=schemas.UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF document, processes it, and stores it in the vector database.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    doc_id = str(uuid.uuid4())
    filename = file.filename
    temp_file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File {filename} (doc_id: {doc_id}) saved to {temp_file_path}")

        chunks = pdf_loader.load_and_split_pdf(temp_file_path)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to process document. No chunks were created.")

        db.add_documents_to_db(chunks, doc_id, filename)
        logger.info(f"Successfully processed and stored doc_id: {doc_id}")
        
        return schemas.UploadResponse(
            filename=filename,
            message=f"Successfully uploaded and processed {len(chunks)} chunks.",
            doc_id=doc_id
        )

    except Exception as e:
        logger.error(f"Error during file upload for {filename}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        file.file.close()


@app.post("/query", 
          tags=["RAG"],
          response_model=schemas.QueryResponse)
async def query_documents(request: schemas.QueryRequest):
    """
    Receives a query, retrieves relevant documents, and generates a response.
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        response_data = rag.get_rag_response(request.query)
        
        source_chunks = [
            schemas.SourceChunk(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in response_data["sources"]
        ]

        return schemas.QueryResponse(
            answer=response_data["answer"],
            sources=source_chunks
        )
        
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/documents", 
         tags=["Documents"],
         response_model=List[schemas.DocumentMetadata])
async def get_all_documents():
    """
    Retrieves metadata for all processed documents. (Placeholder)
    """
    logger.info("Retrieving all document metadata (placeholder)...")
    return [
        schemas.DocumentMetadata(
            doc_id="example-doc-id",
            filename="example.pdf",
            total_pages=10
        )
    ]