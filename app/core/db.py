# File: app/core/db.py

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import logging
from typing import List
from langchain_core.documents import Document

# Set up logging
logger = logging.getLogger(__name__)

# Define the path for the persistent Chroma database
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "data", "chroma_db")
CHROMA_COLLECTION = "rag_collection"

# --- Globals that will be initialized ---
EMBEDDING_MODEL = None
vector_store = None # We still set the global for 'add_documents_to_db'
db_client = None

# --- UPDATED INITIALIZATION FUNCTION ---
def initialize_database():
    """
    Initializes the embedding model and vector database.
    Returns:
        The initialized Chroma vector store, or None if failed.
    """
    global EMBEDDING_MODEL, vector_store, db_client
    
    try:
        logger.info("Loading HuggingFace embedding model...")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None  # <-- Return None on failure

    try:
        logger.info(f"Initializing ChromaDB client at {CHROMA_PERSIST_DIR}...")
        db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        vector_store = Chroma(
            client=db_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=EMBEDDING_MODEL,
        )
        logger.info("ChromaDB client and vector store initialized.")
        return vector_store  # <-- RETURN the new object

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return None  # <-- Return None on failure

def add_documents_to_db(chunks: List[Document], doc_id: str, filename: str):
    """
    Embeds document chunks and adds them to the vector database.
    """
    global vector_store
    if vector_store is None:
        logger.error("Vector store is not initialized. Cannot add documents.")
        return
    
    if not chunks:
        logger.warning(f"No chunks provided for doc_id {doc_id}. Nothing to add.")
        return

    try:
        logger.info(f"Adding {len(chunks)} chunks for doc_id: {doc_id}...")
        
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source"] = filename

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        logger.info(f"Successfully added {len(chunks)} chunks for doc_id: {doc_id}")
    
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")

def get_vector_store() -> Chroma:
    """
    Returns the initialized vector store instance. 
    (Used by rag.py *after* initialization)
    """
    return vector_store