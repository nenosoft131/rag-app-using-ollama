import io
from PyPDF2 import PdfReader
from typing import List, Optional

class PDFService:
    def __init__(self):
        pass
    
    async def extract_text(self, file) -> Optional[str]:
        """
        Extract text from uploaded PDF file.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Extracted text as string, or None if failed
        """
        try:
            content = await file.read()
            pdf_reader = PdfReader(io.BytesIO(content))
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks for better embedding.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap if end < len(text) else end
        
        return chunks