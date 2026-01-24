from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
import uuid

from services.ollama_service import OllamaService
from services.vector_service import VectorService

class RAGState:
    def __init__(self):
        self.messages: List = []
        self.query: str = ""
        self.context: List[str] = []
        self.response: str = ""
        self.session_id: str = ""
        self.sources: List[str] = []

class RAGWorkflow:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.ollama_service = OllamaService()
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for RAG."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("format_response", self._format_response)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    async def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on the query."""
        try:
            # Search for relevant documents
            relevant_docs = self.vector_service.search(state.query, k=3)
            state.context = relevant_docs['context']
            state.sources = relevant_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            state.context = []
            state.sources = []
        
        return state
    
    async def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using Ollama with retrieved context."""
        try:
            # Create context-aware prompt
            # print(state.context)
            context_text = "\n\n".join(state.context) if state.context else "No relevant context found."
            
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer questions. If the context doesn't contain 
            enough information to answer the question, say so politely."""
            
            full_prompt = f"Context: {context_text}\n\nQuestion: {state.query}"
            
            # Generate response using Ollama
            response = await self.ollama_service.chat(
                prompt=full_prompt,
                system_prompt=system_prompt,
                model=getattr(state, 'model', 'llama2')
            )
            
            state.response = response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            state.response = f"Sorry, I encountered an error while processing your request: {str(e)}"
        
        return state
    
    async def _format_response(self, state: RAGState) -> RAGState:
        """Format the final response."""
        # Ensure session_id is set
        if not state.session_id:
            state.session_id = str(uuid.uuid4())
        
        return state
    
    async def process_message(self, message: str, session_id: str = None, model: str = "llama2") -> Dict[str, Any]:
        """Process a message through the RAG workflow."""
        # Create initial state
        state = RAGState()
        state.query = message
        state.session_id = session_id or str(uuid.uuid4())
        state.model = model
        
        # Run the workflow
        try:
            final_state = await self.workflow.ainvoke(state)
            
            return {
                "response": final_state.response,
                "session_id": final_state.session_id,
                "sources": final_state.sources
            }
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "response": f"Error processing message: {str(e)}",
                "session_id": state.session_id,
                "sources": []
            }