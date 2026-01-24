from sentence_transformers import SentenceTransformer
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class VectorService:
    def __init__(self, collection_name: str = "pdf_documents"):
        """
        Initialize vector store with ChromaDB and sentence transformers.
        
        Args:
            collection_name: Name for the ChromaDB collection
        """
        # self.client = chromadb.Client()
        # self.collection_name = collection_name
        # self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        
        from langchain_ollama import OllamaEmbeddings

        # 1️⃣ Setup Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"  # make sure Ollama is running
        )

        
    def add_documents(self, texts: List[str], source_name: str = "unknown") -> None:
        if not texts:
            return

        documents = [
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "chunk_index": idx
                }
            )
            for idx, text in enumerate(texts)
        ]
        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

            # self.vector_store.save_local(self.persist_path)
        except Exception as e:
            raise (f"Error processing document: {str(e)}")
    
    def search(self, query: str, k: int = 3) -> List[str]:
            
        if not query:
            return []
        try:
            retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
            result = retriever.invoke(query)

            context = [doc.page_content for doc in result]
            metadata = [doc.metadata for doc in result]

            return {
                'query': query,
                'context': context,
                'metadata': metadata
            }
        except Exception as e:
            print(f"Error retrieving vector: {e}")
        # # query_embedding = self.embedding_model.encode([query])
        # query_embedding = self.embeddings.embed_documents([query])
        
        # results = self.collection.query(
        #     query_embeddings=query_embedding,
        #     n_results=k
        # )
        
        # return results['documents'][0] if results['documents'] else []
    
    def get_vector_size(self) -> int:
        """Get the number of documents in the collection."""
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal
    
    # def clear_all_collection(self,collection):
    #     all_ids = collection.get(include=["ids"])["ids"]
    #     if all_ids:
    #         collection.delete(ids=all_ids)
    #     print("Collection cleared")
    
    def clear_vector_store(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Clear collection safely
            self.vector_store = None
        except Exception as e:
            print(f"Error clearing collection: {e}")
            
    