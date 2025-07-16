import logging
from typing import Dict, Any, Optional
from .abstract_agent import AbstractAgent
from utils.vector_store import VectorStore
from utils.embedding_utils import DocumentProcessor
from config import config

class RagBot(AbstractAgent):
    """
    RAG Bot that uses pgvector for document storage and retrieval.
    """
    
    def __init__(self, name: str = "RAG Bot", description: str = "Retrieval-Augmented Generation Bot with Vector Database", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, description, config)
        
        # Initialize components
        self.vector_store = VectorStore("rag_documents")
        self.doc_processor = DocumentProcessor()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the RAG bot with vector database."""
        self.logger.info("Initializing RAG bot with vector database...")
        
        try:
            # Initialize document processor (embeddings)
            doc_processor_success = await self.doc_processor.initialize()
            if not doc_processor_success:
                self.logger.error("Failed to initialize document processor")
                return False
            
            # Initialize vector store (pgvector)
            vector_store_success = await self.vector_store.initialize()
            if not vector_store_success:
                self.logger.error("Failed to initialize vector store")
                return False
            
            # Load sample content if database is empty
            doc_count = await self.vector_store.get_document_count()
            if doc_count == 0:
                await self._load_sample_content()
            
            self.initialized = True
            self.logger.info(f"RAG bot initialized successfully with {await self.vector_store.get_document_count()} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG bot: {e}")
            return False
    
    async def invoke(self, message: str) -> str:
        """Invoke the RAG bot to process queries using vector search."""
        self.logger.info(f"RAG bot processing message: {message}")
        
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return "❌ RAG bot initialization failed. Please check database connections."
        
        try:
            # Generate query embedding
            query_embedding = self.doc_processor.embedder.embed_text(message)
            
            # Perform similarity search
            similar_docs = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=3,
                threshold=config.SIMILARITY_THRESHOLD
            )
            
            # Debug: log what we found
            self.logger.info(f"Debug similarity search: threshold={config.SIMILARITY_THRESHOLD}, found {len(similar_docs)} docs")
            for i, doc in enumerate(similar_docs):
                similarity = doc.get('similarity', 0)
                content_preview = doc.get('content', '')[:50] + '...'
                self.logger.info(f"  Doc {i}: similarity={similarity:.4f}, content='{content_preview}'")
            
            if not similar_docs:
                return f"🔍 No relevant documents found in my knowledge base for: '{message}'. The database contains {await self.vector_store.get_document_count()} documents. You might want to try a different query or check if content has been loaded."
            
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(similar_docs, 1):
                similarity = doc.get('similarity', 0)
                content = doc.get('content', '')
                context_parts.append(f"[Document {i} - Similarity: {similarity:.3f}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response based on retrieved context
            response = f"""📚 **RAG Bot Response** (Found {len(similar_docs)} relevant documents)

**Your Question:** {message}

**Based on my knowledge base:**
{self._generate_contextual_response(message, context)}

**Sources:**
{self._format_sources(similar_docs)}
"""
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in RAG bot invoke: {e}")
            return f"❌ Error processing your query: {str(e)}"
    
    async def _load_sample_content(self):
        """Load sample Healthee content into the vector database."""
        try:
            # Read the healthee.md file content
            import os
            healthee_file = os.path.join(os.path.dirname(__file__), '..', 'healthee.md')
            
            if os.path.exists(healthee_file):
                with open(healthee_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Process the content into documents with embeddings
                documents = await self.doc_processor.process_text(
                    content, 
                    metadata={
                        'source': 'healthee.md',
                        'type': 'company_info',
                        'domain': 'healthcare_benefits'
                    }
                )
                
                # Store in vector database
                if documents:
                    success = await self.vector_store.add_documents(documents)
                    if success:
                        self.logger.info(f"Loaded {len(documents)} documents from healthee.md")
                    else:
                        self.logger.error("Failed to store sample documents")
                else:
                    self.logger.warning("No documents generated from healthee.md")
            else:
                self.logger.warning("healthee.md file not found, skipping sample content loading")
                
        except Exception as e:
            self.logger.error(f"Error loading sample content: {e}")
    
    def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate a contextual response based on retrieved documents."""
        # Simple response generation based on context
        # In a full implementation, this could use an LLM for better responses
        
        if "healthee" in query.lower():
            if "features" in query.lower() or "what" in query.lower():
                return "Based on the retrieved documents, Healthee offers several key features including AI-powered benefits assistance, plan navigation & comparison, and integrated wellness tools."
            elif "serve" in query.lower() or "who" in query.lower():
                return "According to my knowledge base, Healthee serves employees who need help understanding their health benefits, HR teams looking to reduce administrative burden, and organizations wanting to improve benefits utilization."
            elif "zoe" in query.lower():
                return "Zoe is Healthee's AI-powered virtual assistant that provides real-time, personalized answers about health insurance coverage, benefits, and healthcare decisions."
        
        # Fallback: extract key information from context
        sentences = context.split('.')[:3]  # Get first few sentences
        return " ".join(sentences).strip() + "."
    
    def _format_sources(self, documents) -> str:
        """Format document sources for display."""
        sources = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            similarity = doc.get('similarity', 0)
            sources.append(f"• Document {i}: {source} (Relevance: {similarity:.1%})")
        
        return "\n".join(sources)
    
    async def add_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add new content to the RAG knowledge base."""
        if not self.initialized:
            return False
        
        try:
            documents = await self.doc_processor.process_text(content, metadata)
            if documents:
                success = await self.vector_store.add_documents(documents)
                self.logger.info(f"Added {len(documents)} new documents to knowledge base")
                return success
            return False
        except Exception as e:
            self.logger.error(f"Error adding content: {e}")
            return False
    
    async def search_knowledge_base(self, query: str, k: int = 5) -> list:
        """Search the knowledge base for relevant documents."""
        if not self.initialized:
            return []
        
        try:
            query_embedding = self.doc_processor.embedder.embed_text(query)
            return await self.vector_store.similarity_search(
                query_embedding, 
                k=k, 
                threshold=config.SIMILARITY_THRESHOLD
            )
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get RAG bot information."""
        info = super().get_info()
        info.update({
            "initialized": self.initialized,
            "database_type": "pgvector",
            "embedding_model": config.EMBEDDING_MODEL,
            "capabilities": ["vector_search", "document_retrieval", "contextual_responses", "knowledge_base_management"]
        })
        return info
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG bot statistics."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            doc_count = await self.vector_store.get_document_count()
            vector_store_healthy = await self.vector_store.health_check()
            
            return {
                "document_count": doc_count,
                "vector_store_healthy": vector_store_healthy,
                "embedding_dimension": self.doc_processor.embedder.embedding_dim,
                "similarity_threshold": config.SIMILARITY_THRESHOLD
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if the RAG bot is healthy."""
        if not self.initialized:
            return False
        
        try:
            return await self.vector_store.health_check()
        except Exception:
            return False
    
    async def close(self):
        """Close RAG bot resources."""
        if self.vector_store:
            await self.vector_store.close()
        self.initialized = False