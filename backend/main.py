from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv

from services.rag_workflow import RAGWorkflow
from services.pdf_service import PDFService
from services.vector_service import VectorService

load_dotenv()

app = FastAPI(title="PDF RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: str = "llama2"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []

class DocumentResponse(BaseModel):
    message: str
    document_count: int

pdf_service = PDFService()
vector_service = VectorService()
rag_workflow = RAGWorkflow(vector_service=vector_service)

@app.get("/")
async def root():
    return {"message": "PDF RAG API is running"}

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        text = await pdf_service.extract_text(file)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
        
        chunks = pdf_service.chunk_text(text)
        vector_service.add_documents(chunks, file.filename)
        
        return DocumentResponse(
            message=f"Successfully processed {file.filename}",
            document_count=vector_service.get_collection_size()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system."""
    try:
        response = await rag_workflow.process_message(
            message=request.message,
            session_id=request.session_id,
            model=request.model
        )
        
        return ChatResponse(
            response=response["response"],
            session_id=response["session_id"],
            sources=response.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get information about stored documents."""
    return {
        "document_count": vector_service.get_collection_size(),
        "available_models": ["llama2", "mistral", "codellama"]
    }

@app.delete("/documents")
async def clear_documents():
    """Clear all stored documents."""
    vector_service.clear_collection()
    return {"message": "All documents cleared successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )