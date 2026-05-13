# RAG App using Ollama

A locally-running PDF question-answering system built with an agentic RAG pipeline. Upload a PDF, ask questions, and get answers grounded in the document — all running on your machine via Ollama (no cloud API required).

## Architecture

```
frontend/          (Streamlit)
    ↕ HTTP
backend/           (FastAPI)
    ↕
LangGraph Pipeline → Ollama (LLM + Embeddings) + FAISS (Vector Store) + SQLite (Memory + Stats)
```

### Backend (FastAPI + LangGraph)

- **FastAPI** — RESTful API server with structured error responses and 503 handling for Ollama downtime
- **LangGraph** — agentic pipeline with conditional routing and retry loops
- **FAISS** — in-memory vector store for document embeddings
- **Ollama** — local LLM (llama2 / mistral / codellama) and `nomic-embed-text` embeddings
- **SQLite** — persistent conversation memory (LangGraph `SqliteSaver`) and persistent pipeline stats

### Frontend (Streamlit)

- **Chat page** — PDF upload, model selector, multi-session chat with source viewing
- **Dashboard page** — live charts, pipeline metrics, recent queries table, filtered execution logs

### Chat Page

![Chat Page](images/app.png)

### Live Dashboard

![Dashboard Overview](images/dash_1.png)

![Dashboard Charts](images/dash_2.png)

![Dashboard Logs](images/dash_3.png)

## Agentic Pipeline

Queries run through a 5-node LangGraph graph with dynamic routing and self-correction:

```
Router → Retrieve → Relevance Grader ⟳(retry) → Generator → Hallucination Grader ⟳(regenerate) → END
         ↘ (general query) ↗
```

| Node | Role |
|---|---|
| **Router** | Classifies the query: needs document retrieval or can answer directly |
| **Retrieve** | Fetches top-4 chunks from FAISS using semantic similarity |
| **Relevance Grader** | Filters out chunks not relevant to the query; retries retrieval if none pass |
| **Generator** | Produces an answer using the filtered context |
| **Hallucination Grader** | Verifies the answer is grounded in retrieved docs; triggers regeneration if not |

## Features

- PDF document upload and processing
- Semantic search with vector embeddings
- Multi-session chat with persistent conversation history (SQLite)
- Agentic self-correction — relevance grading and hallucination checking with automatic retries
- Support for multiple Ollama models (llama2, mistral, codellama)
- **Retry logic** — LLM calls retry up to 3 times with exponential backoff (via `tenacity`)
- **Timeout handling** — 60-second HTTP timeout on all Ollama calls
- **Structured error responses** — 503 when Ollama is unreachable, 400/500 for other errors
- **Persistent stats** — query metrics and history survive backend restarts (stored in SQLite)
- **Live dashboard** with charts, recent query table, log search/filter, and pause control
- Fully local — no external API calls

## Dashboard

The dashboard page (`/Dashboard`) auto-refreshes and shows:

| Section | What it shows |
|---|---|
| **Overview KPIs** | Total queries, retrieval %, general %, success rate, avg latency, errors |
| **Route Distribution** | Pie chart of retrieval vs general query split |
| **Query Volume Over Time** | Bar chart bucketed by minute |
| **Model Usage** | Bar chart of queries per model |
| **Latency Trend** | Line chart of response time across last 30 queries |
| **Recent Queries** | Table with time, query text, route, model, retries, latency, success |
| **Agent Pipeline** | Call counts per node |
| **Execution Log** | Color-coded logs with keyword search and level filter (All / Error / Retry / Success / Step) |

The refresh control has an **Auto-refresh** toggle, a **Pause** toggle (freeze the view without disabling refresh), and a configurable interval (3 / 5 / 10 seconds).

## Setup

### Prerequisites

1. **Ollama** — install and pull a model:

   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve

   ollama pull llama2
   ollama pull nomic-embed-text   # required for embeddings
   ```

2. **Python 3.9+**

### Installation

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
pip install -r requirements.txt
```

## Running

### 1. Start the Backend

```bash
cd backend
python main.py
# API available at http://localhost:8000
```

### 2. Start the Frontend

```bash
cd frontend
streamlit run app.py
# UI available at http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload and process a PDF |
| `POST` | `/chat` | Send a message |
| `GET` | `/documents` | Get document store info |
| `DELETE` | `/documents` | Clear all documents |
| `GET` | `/stats` | Pipeline metrics, recent queries, and logs |

### Examples

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?", "model": "llama2"}'
```

### Error Responses

All errors return structured JSON:

```json
{ "error": "human-readable message", "detail": "raw exception detail" }
```

| Status | Meaning |
|---|---|
| `400` | Invalid input (e.g. non-PDF file) |
| `503` | Ollama is unreachable |
| `500` | Internal server error |

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL (backend) |
| `API_BASE_URL` | `http://localhost:8000` | Backend URL (frontend) |

### Default Settings

- Embedding model: `nomic-embed-text`
- Chunk size: 1000 characters, 200-character overlap
- Retrieval top-k: 4 chunks
- Default LLM: `llama2`
- LLM call timeout: 60 seconds
- LLM retry attempts: 3 (exponential backoff: 2s → 4s → 8s)

## Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI app, routes, and error handlers
│   ├── stats.py                   # Persistent pipeline metrics (SQLite-backed)
│   ├── services/
│   │   ├── rag_workflow.py        # LangGraph agentic pipeline + query timing
│   │   ├── pdf_service.py         # PDF parsing and chunking
│   │   ├── vector_service.py      # FAISS vector store wrapper
│   │   └── ollama_service.py      # Ollama client with retry + timeout
│   ├── evaluation/
│   │   ├── evaluate.py            # RAGAS evaluation runner
│   │   └── evaluation_dataset.py  # Test dataset
│   └── requirements.txt
├── frontend/
│   ├── app.py                     # Chat page (Streamlit)
│   ├── pages/
│   │   └── 1_Dashboard.py         # Live dashboard with charts and log viewer
│   ├── api_client.py              # HTTP client for backend
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

## Docker

```bash
docker-compose up --build
```

## Troubleshooting

- **Ollama not connecting** — ensure `ollama serve` is running on port 11434; the API returns 503 with a clear message when it's down
- **Model not found** — run `ollama pull <model-name>` before starting the backend
- **LLM calls timing out** — default timeout is 60s; for slow hardware consider increasing `TIMEOUT` in `ollama_service.py`
- **API connection error in UI** — ensure the backend is running on port 8000
- **Dashboard shows no charts** — send at least one message in the Chat page first to populate query data
