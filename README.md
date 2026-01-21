#RAG app using Ollama

A modern RAG (Retrieval-Augmented Generation) application with separated frontend and backend architecture using LangGraph, FastAPI, and Streamlit.

## Architecture

### Backend (FastAPI + LangGraph)

- **FastAPI**: RESTful API server
- **LangGraph**: Workflow orchestration for RAG pipeline
- **ChromaDB**: Vector storage for document embeddings
- **Ollama**: LLM integration for response generation
- **Sentence Transformers**: Text embeddings

### Frontend (Streamlit)

- **Streamlit**: Web interface for document upload and chat
- **API Client**: HTTP client for backend communication

## Features

- 📄 PDF document upload and processing
- 🔍 Semantic search with vector embeddings
- 💬 Chat interface with context-aware responses
- 🤖 Integration with Ollama models (Llama2)
- 🔄 LangGraph workflow for RAG pipeline
- 🌐 Separated frontend/backend architecture
- 📊 Real-time document chunking and indexing
- 🎯 Session-based conversation continuity

## Setup

### Prerequisites

1. **Ollama Installation**: Install and start Ollama

   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Start Ollama server
   ollama serve

   # Pull a model
   ollama pull llama2
   ```

2. **Python Environment**: Python 3.9+

### Installation

1. **Backend Setup**:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Start the Backend

```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

### 2. Start the Frontend

```bash
cd frontend
streamlit run app.py
```

The frontend will start on `http://localhost:8501`

## API Endpoints

### Backend Endpoints

- `GET /`: Health check
- `POST /upload`: Upload and process PDF documents
- `POST /chat`: Chat with the RAG system
- `GET /documents`: Get document information
- `DELETE /documents`: Clear all documents

### Usage Examples

#### Upload Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### Chat

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is this document about?",
    "model": "llama2"
  }'
```

## LangGraph Workflow

The RAG pipeline is implemented using LangGraph with the following nodes:

1. **Retrieve**: Search for relevant documents using vector similarity
2. **Generate**: Create response using Ollama with retrieved context
3. **Format**: Format the final response with sources

## Configuration

### Environment Variables

Backend:

- `OLLAMA_HOST`: Ollama server host (default: `http://localhost:11434`)

Frontend:

- `API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)

### Default Settings

- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: 1000 characters with 200 character overlap
- Search results: 3 most relevant chunks
- Default LLM: `llama2`

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── services/
│   │   ├── rag_workflow.py     # LangGraph RAG workflow
│   │   ├── pdf_service.py      # PDF processing
│   │   ├── vector_service.py   # Vector storage
│   │   └── ollama_service.py   # Ollama integration
│   └── requirements.txt
├── frontend/
│   ├── app.py                  # Streamlit application
│   ├── api_client.py           # API client
│   └── requirements.txt
└── README.md
```

## Troubleshooting

### Backend Issues

- **Ollama Connection**: Ensure Ollama is running on `localhost:11434`
- **Model Not Found**: The app will automatically pull models if available
- **Vector Storage**: ChromaDB runs in-memory by default

### Frontend Issues

- **API Connection**: Ensure backend is running on `localhost:8000`
- **CORS Issues**: Backend includes CORS middleware for frontend access
- **File Upload**: Check file size limits and PDF format

### Development Tips

- Use `uvicorn main:app --reload` for backend development
- Use `streamlit run app.py --server.reload` for frontend development
- Check browser console for API errors
