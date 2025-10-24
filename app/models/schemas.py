# File: app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    """
    The request model for the /query endpoint.
    """
    query: str

class SourceChunk(BaseModel):
    """
    A model to represent a single retrieved source chunk.
    """
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    """
    The response model for the /query endpoint.
    """
    answer: str
    sources: List[SourceChunk]

class UploadResponse(BaseModel):
    """
    The response model for the /upload endpoint.
    """
    filename: str
    message: str
    doc_id: str

class DocumentMetadata(BaseModel):
    """
    The response model for the /documents/{doc_id} endpoint.
    """
    doc_id: str
    filename: str
    total_pages: int