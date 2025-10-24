# File: tests/test_rag.py

import pytest
import httpx  # This is like 'requests' but for async apps
import os
import time

# Base URL of our running FastAPI app
BASE_URL = "http://127.0.0.1:8000"

# --- Test Data ---
# Create a dummy PDF for testing so we don't need a real one
TEST_PDF_NAME = "test_document.pdf"

@pytest.fixture(scope="module")
def setup_test_pdf():
    """
    Creates a simple, dummy PDF file for testing uploads.
    We're doing this so the test doesn't depend on an external file.
    """
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(TEST_PDF_NAME)
        c.drawString(100, 750, "This is a test document for the RAG API.")
        c.drawString(100, 735, "The main topic is about testing and nothing else.")
        c.save()
        print(f"Created dummy PDF: {TEST_PDF_NAME}")
    except ImportError:
        pytest.skip("reportlab not installed. Skipping upload test.")

    yield
    
    # Teardown: Remove the dummy PDF after tests are done
    if os.path.exists(TEST_PDF_NAME):
        os.remove(TEST_PDF_NAME)
        print(f"\nRemoved dummy PDF: {TEST_PDF_NAME}")

@pytest.mark.asyncio
async def test_server_health():
    """
    Tests if the server is running and the / endpoint is reachable.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL + "/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "RAG API is running."}

@pytest.mark.asyncio
async def test_upload_and_query(setup_test_pdf):
    """
    An integration test that uploads a document and then queries it.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        # --- 1. Test Upload ---
        file_to_upload = {'file': (TEST_PDF_NAME, open(TEST_PDF_NAME, 'rb'), 'application/pdf')}
        
        upload_response = await client.post(BASE_URL + "/upload", files=file_to_upload)
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        assert upload_data["filename"] == TEST_PDF_NAME
        assert "Successfully uploaded and processed" in upload_data["message"]

        # Give the server a moment to make sure embeddings are fully committed
        time.sleep(2)

        # --- 2. Test Query ---
        query_payload = {"query": "What is the main topic of this document?"}
        
        query_response = await client.post(BASE_URL + "/query", json=query_payload)
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        assert "answer" in query_data
        assert "sources" in query_data
        
        # Check if the answer is relevant
        answer = query_data["answer"].lower()
        print(f"Test Answer: {answer}")
        assert "test" in answer or "testing" in answer