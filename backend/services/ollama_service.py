import ollama
from typing import Optional

class OllamaService:
    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        """
        Initialize Ollama service.
        
        Args:
            model: Default model to use
            host: Ollama server host
        """
        self.model = model
        self.host = host
        
    async def chat(self, prompt: str, system_prompt: Optional[str] = None, model: str = "llama2") -> str:
        """
        Generate response using Ollama model.
        
        Args:
            prompt: User question/prompt
            system_prompt: Optional system prompt
            model: Model to use for generation
            
        Returns:
            Generated response
        """
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer questions. If the context doesn't contain 
            enough information to answer the question, say so politely."""
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=model or self.model,
                messages=messages
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def list_models(self) -> list:
        """List available models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            ollama.pull(model_name)
            return True
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False
    
    def set_model(self, model: str) -> None:
        """Set the default model to use."""
        self.model = model