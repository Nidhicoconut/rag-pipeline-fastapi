# Full-Stack RAG API

This project is a complete Retrieval-Augmented Generation (RAG) pipeline built with FastAPI and containerized with Docker. It allows users to upload PDF documents and ask questions based on their content.

This application uses:
- **FastAPI** for the REST API
- **LangChain** to orchestrate the RAG pipeline
- **ChromaDB** as a local, persistent vector database
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for creating embeddings
- **Google Gemini** for the LLM generation
- **Docker** & **Docker Compose** for containerization

---

## Quick Start

This entire application is containerized. The only prerequisites are **Docker Desktop** and a **Google API Key**.

### 1. Set Up Your Environment

Before you begin, you must provide your Google API key.

1.  Find the file named `.env.example` in this folder.
2.  Create a **new file** in this same folder and name it `.env`
3.  Copy the contents of `.env.example` into your new `.env` file and paste your API key:

    ```bash
    # File: .env
    GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_GOES_HERE"
    ```

### 2. Build and Run the Application

With Docker Desktop running, open your terminal in the project's root folder (the one containing `docker-compose.yml`) and run:

```bash
docker-compose up --build
