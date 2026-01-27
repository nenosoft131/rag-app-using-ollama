import pytest
import io
from pdf_service import PDFService  # assuming your class is in pdf_service.py
from PyPDF2 import PdfWriter


def create_test_pdf(pages_text):
    pdf_bytes = io.BytesIO()
    pdf_writer = PdfWriter()
    
    for text in pages_text:
        pdf_writer.add_blank_page(width=72, height=72)
    pdf_writer.write(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

@pytest.mark.asyncio
async def test_extract_text_returns_none_on_empty_file():
    service = PDFService()
    class DummyFile:
        async def read(self):
            return b""  # empty PDF
    file = DummyFile()
    
    text = await service.extract_text(file)
    assert text is None

@pytest.mark.asyncio
async def test_extract_text_returns_text(monkeypatch):
    service = PDFService()
    
    # Mock PdfReader to return pages with extract_text
    class DummyPage:
        def extract_text(self):
            return "Hello World"
    
    class DummyPdfReader:
        pages = [DummyPage(), DummyPage()]
    
    monkeypatch.setattr("pdf_service.PdfReader", lambda f: DummyPdfReader())
    
    class DummyFile:
        async def read(self):
            return b"%PDF-1.4 dummy content"
    
    file = DummyFile()
    text = await service.extract_text(file)
    assert "Hello World" in text
    assert text.count("Hello World") == 2

# ----------------------
# Tests for chunk_text
# ----------------------
def test_chunk_text_empty():
    service = PDFService()
    chunks = service.chunk_text("")
    assert chunks == []

def test_chunk_text_chunking():
    service = PDFService()
    text = "a" * 2500  # 2500 characters
    chunks = service.chunk_text(text, chunk_size=1000, overlap=200)
    
    # Should produce 3 chunks: 0-1000, 800-1800, 1600-2500
    assert len(chunks) == 3
    assert chunks[0] == "a" * 1000
    assert chunks[1] == "a" * 1000
    assert chunks[2] == "a" * 900
