import requests
from typing import Dict, Any, Optional, List
import streamlit as st

class RAGAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
       
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def upload_document(self, file) -> Dict[str, Any]:
        
        url = f"{self.base_url}/upload"
        
        # Prepare file for upload
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        try:
            response = self.session.post(url, files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to upload document: {str(e)}")
    
    def chat(self, message: str, session_id: Optional[str] = None, model: str = "llama2") -> Dict[str, Any]:

        url = f"{self.base_url}/chat"
        
        payload = {
            "message": message,
            "model": model
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to send chat message: {str(e)}")
    
    def get_documents(self) -> Dict[str, Any]:
       
        url = f"{self.base_url}/documents"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get documents: {str(e)}")
    
    def clear_documents(self) -> Dict[str, Any]:
        
        url = f"{self.base_url}/documents"
        
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to clear documents: {str(e)}")
    
    def health_check(self) -> bool:
     
        try:
            url = f"{self.base_url}/"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        try:
            docs_info = self.get_documents()
            return docs_info.get("available_models", ["llama2", "mistral", "codellama"])
        except Exception:
            return ["llama2", "mistral", "codellama"]