import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv

from services.rag_workflow import RAGWorkflow
from services.pdf_service import PDFService
from services.vector_service import VectorService
from stats import stats
import chat_store

load_dotenv()

app = FastAPI(title="PDF RAG API", version="1.0.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    msg = str(exc)
    if any(k in msg.lower() for k in ("connection refused", "connect call failed", "ollama")):
        return JSONResponse(
            status_code=503,
            content={"error": "Ollama service unavailable. Ensure Ollama is running on the configured host.", "detail": msg},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error.", "detail": msg},
    )

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
    # sources: List[str] = []


class DocumentResponse(BaseModel):
    message: str
    document_count: int


pdf_service = PDFService()
vector_service = VectorService()
rag_workflow = RAGWorkflow(vector_service=vector_service)


@app.get("/")
async def root():
    return {"message": "RAG API is running"}


@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        text = await pdf_service.extract_text(file)
        if not text:
            raise HTTPException(
                status_code=400, detail="Failed to extract text from PDF"
            )

        chunks = pdf_service.chunk_text(text)
        vector_service.add_documents(chunks, file.filename)
        stats.inc("docs_uploaded")

        return DocumentResponse(
            message=f"Successfully processed {file.filename}",
            document_count=vector_service.get_vector_size(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat with the RAG system."""
    stats.inc("total_queries")
    try:
        response = rag_workflow.process_message(
            message=request.message, session_id=request.session_id, model=request.model
        )
        sid = response["session_id"]
        chat_store.save_message(sid, "user", request.message)
        chat_store.save_message(sid, "assistant", response["response"])
        return ChatResponse(response=response["response"], session_id=sid)
    except Exception as e:
        stats.inc("errors")
        msg = str(e)
        if any(k in msg.lower() for k in ("connection refused", "connect call failed", "ollama")):
            raise HTTPException(status_code=503, detail="Ollama service unavailable. Ensure Ollama is running.")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {msg}")


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    """Return the full message history for a session."""
    return {"session_id": session_id, "messages": chat_store.get_history(session_id)}


@app.get("/chat/sessions")
def get_sessions():
    """Return all session IDs that have history."""
    return {"sessions": chat_store.get_all_sessions()}


@app.get("/stats")
def get_stats():
    """Live agent execution statistics."""
    data = stats.snapshot()
    data["document_count"] = vector_service.get_vector_size()
    return data


@app.get("/documents")
async def get_documents():
    """Get information about stored documents."""
    return {
        "document_count": vector_service.get_vector_size(),
        "available_models": ["llama2", "mistral", "codellama"],
    }


@app.delete("/documents")
async def clear_documents():
    """Clear all stored documents."""
    vector_service.clear_vector_store()
    return {"message": "All documents cleared successfully"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
