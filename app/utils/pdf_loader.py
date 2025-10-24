# File: app/utils/pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_split_pdf(file_path: str):
    """
    Loads a PDF from the given file path and splits it into chunks.
    
    Args:
        file_path: The temporary path to the uploaded PDF file.

    Returns:
        A list of Document objects (chunks).
    """
    try:
        # 1. Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No documents loaded from {file_path}. It might be empty or corrupt.")
            return []

        # 2. Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Loaded and split {file_path} into {len(chunks)} chunks.")
        
        return chunks

    except Exception as e:
        logger.error(f"Error loading or splitting PDF {file_path}: {e}")
        return []