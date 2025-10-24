# File: app/core/rag.py

import os
import logging
from . import db

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma  # <-- Import for type hint

# Set up logging
logger = logging.getLogger(__name__)

# --- 1. Initialize the LLM ---
llm = None
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    logger.info("Google Generative AI model loaded.")
except Exception as e:
    logger.error(f"Failed to load Google Generative AI model: {e}")
    llm = None

# --- 2. Define the Prompt Template ---
PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUESTION:
{question}

Based on the context provided, please provide a concise and relevant answer to the question. If the context does not contain the answer, state that the information is not available in the provided documents.
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# --- 3. Global variables for the chain and retriever ---
rag_chain = None
retriever = None

# --- 4. Helper function to format documents ---
def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 5. UPDATED Initialization Function ---
def initialize_rag_chain(vector_store: Chroma):  # <-- ACCEPT ARGUMENT HERE
    """
    Initializes the RAG chain with the provided vector store.
    """
    global rag_chain, retriever, llm
    
    if llm is None:
        logger.error("LLM is not initialized. RAG chain cannot be created.")
        return
    
    if vector_store is None:  # <-- THIS IS THE FIX  # <-- Check the argument
        logger.error("Vector store object is invalid. Cannot create retriever.")
        return

    try:
        retriever = vector_store.as_retriever(k=4)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain created successfully.")

    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")

# --- 6. Public function to be called by the API ---
def get_rag_response(query: str):
    """
    Gets a query, runs the RAG chain, and returns the response.
    """
    global rag_chain, retriever

    if not rag_chain or not retriever:
        logger.error("RAG chain or retriever is not initialized.")
        return {"answer": "Error: RAG system is not ready. Check server logs.", "sources": []}

    try:
        relevant_docs = retriever.invoke(query)
        answer = rag_chain.invoke(query)
        return {"answer": answer, "sources": relevant_docs}
    
    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {e}")
        return {"answer": f"An error occurred: {e}", "sources": []}