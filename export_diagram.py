from fpdf import FPDF
from fpdf.enums import XPos, YPos

DIAGRAM = """\
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER / CLI                          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FRONTEND  (Streamlit :8501)                     │
│                                                                     │
│   ┌─────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│   │  PDF Upload │   │   Chat Interface  │   │  Sidebar          │   │
│   │  Widget     │   │  (messages +      │   │  - Model select   │   │
│   └──────┬──────┘   │   sources)        │   │  - Thread history │   │
│          │          └────────┬──────────┘   │  - Clear docs     │   │
│          │                   │              └───────────────────┘   │
│          └──────────┬────────┘                                      │
│                     │  RAGAPIClient (requests)                      │
└─────────────────────┼───────────────────────────────────────────────┘
                      │ HTTP  (localhost:8000)
          ┌───────────┼──────────────────────┐
          │           │                      │
     POST /upload  POST /chat         GET|DELETE /documents
          │           │                      │
┌─────────▼───────────▼──────────────────────▼─────────────────────────┐
│                      BACKEND  (FastAPI :8000)                         │
│                                                                       │
│  ┌─────────────────┐          ┌────────────────────────────────────┐  │
│  │   PDFService    │          │         RAGWorkflow                │  │
│  │                 │          │        (LangGraph)                 │  │
│  │ • extract_text  │          │                                    │  │
│  │   (PyPDF2)      │          │  ┌──────────┐                      │  │
│  │                 │          │  │ retrieve │                      │  │
│  │ • chunk_text    │          │  │  node    │                      │  │
│  │   1000 chars /  │          │  └────┬─────┘                      │  │
│  │   200 overlap   │          │       │                            │  │
│  └────────┬────────┘          │  ┌────▼─────┐                      │  │
│           │ chunks            │  │ generate │                      │  │
│           │                   │  │  node    │                      │  │
│  ┌────────▼────────┐          │  └────┬─────┘                      │  │
│  │  VectorService  │<--search--│       │                            │  │
│  │                 │          │  ┌────▼──────────┐                 │  │
│  │ • FAISS index   │          │  │ format_response│                │  │
│  │   (in-memory)   │          │  │    node        │                │  │
│  │                 │          │  └───────────────┘                 │  │
│  │ • OllamaEmbeds  │          │                                    │  │
│  │   (embed query  │          │  SqliteSaver → chatbot.db          │  │
│  │    + chunks)    │          │  (conversation history per         │  │
│  └─────────────────┘          │   thread_id)                       │  │
│                               └────────────────┬───────────────────┘  │
└────────────────────────────────────────────────┼──────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                     OLLAMA  (localhost:11434)                      │
│                                                                    │
│   ┌──────────────────────┐     ┌─────────────────────────────┐    │
│   │  nomic-embed-text    │     │  llama2 / mistral /         │    │
│   │  (embeddings for     │     │  codellama                  │    │
│   │   upload + search)   │     │  (chat completion)          │    │
│   └──────────────────────┘     └─────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE                                    │
│                                                                    │
│   chatbot.db  (SQLite)          FAISS index                        │
│   └─ LangGraph checkpoints      └─ in-memory only                 │
│      (survives restart)            (lost on restart)               │
└────────────────────────────────────────────────────────────────────┘

Request paths:
  Upload : Streamlit → POST /upload → PDFService (extract+chunk)
             → VectorService (embed+index)
  Chat   : Streamlit → POST /chat → RAGWorkflow
             → [retrieve → VectorService → Ollama embed]
             → [generate → OllamaService → Ollama LLM]
             → response
"""


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "RAG App - System Architecture", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


pdf = PDF(orientation="L", unit="mm", format="A4")
pdf.set_margins(12, 14, 12)
pdf.set_auto_page_break(auto=True, margin=14)

pdf.add_font("Mono", fname="/System/Library/Fonts/Supplemental/Andale Mono.ttf")

pdf.add_page()

pdf.set_font("Mono", size=7.2)
pdf.set_text_color(30, 30, 30)

for line in DIAGRAM.splitlines():
    pdf.cell(0, 3.8, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

out = "architecture_diagram.pdf"
pdf.output(out)
print(f"Saved: {out}")
