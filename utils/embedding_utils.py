"""
Embedding and text processing utilities.
Handles text embeddings, chunking, and document processing.
"""

import logging
import hashlib
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from config import config
from utils.vector_store import Document

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Handles text embedding generation using sentence transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the text embedder."""
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.embedding_dim = config.EMBEDDING_DIMENSION
        
    def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Verify embedding dimension
            test_embedding = self.model.encode("test")
            actual_dim = len(test_embedding)
            
            if actual_dim != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {actual_dim}")
                self.embedding_dim = actual_dim
            
            logger.info(f"Embedding model loaded successfully (dimension: {self.embedding_dim})")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            return False
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embeddings = self.model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

class TextChunker:
    """Handles text chunking with overlap."""
    
    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """Initialize the text chunker."""
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split text into chunks
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk and we're not at the end of the text,
            # try to find a good breaking point (sentence or paragraph end)
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                search_text = text[search_start:end + 100]
                
                # Find the last sentence ending
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', search_text)]
                if sentence_ends:
                    last_sentence_end = sentence_ends[-1]
                    end = search_start + last_sentence_end
            
            # Extract the chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Create document ID
                doc_id = self._generate_chunk_id(chunk_text, chunk_id)
                
                # Create chunk metadata
                chunk_metadata = {
                    **metadata,
                    'chunk_id': chunk_id,
                    'chunk_start': start,
                    'chunk_end': end,
                    'chunk_size': len(chunk_text)
                }
                
                # Create document
                doc = Document(
                    id=doc_id,
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                
                chunks.append(doc)
                chunk_id += 1
            
            # Move to next chunk with overlap
            start = end - self.overlap
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def _generate_chunk_id(self, text: str, chunk_num: int) -> str:
        """Generate a unique ID for a text chunk."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"chunk_{chunk_num}_{text_hash}"

class DocumentProcessor:
    """Processes documents for RAG and Graph RAG systems."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.embedder = TextEmbedder()
        self.chunker = TextChunker()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the document processor."""
        try:
            self.initialized = self.embedder.initialize()
            logger.info("Document processor initialized successfully")
            return self.initialized
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}")
            return False
    
    async def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process text into documents with embeddings."""
        if not self.initialized:
            raise RuntimeError("Document processor not initialized")
        
        try:
            # Chunk the text
            documents = self.chunker.chunk_text(text, metadata)
            
            if not documents:
                return []
            
            # Generate embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.embedder.embed_batch(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            logger.info(f"Processed text into {len(documents)} documents with embeddings")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from text (simple implementation)."""
        # This is a simple implementation - could be enhanced with NLP libraries
        concepts = []
        
        # Extract potential concepts (capitalized phrases, technical terms)
        concept_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:AI|ML|API|UI|UX|DB)\b',  # Tech terms
        ]
        
        concept_set = set()
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            concept_set.update(matches)
        
        # Convert to concept objects
        for i, concept_text in enumerate(concept_set):
            if len(concept_text) > 2:  # Filter out very short matches
                concept_id = f"concept_{hashlib.md5(concept_text.encode()).hexdigest()[:8]}"
                concepts.append({
                    'id': concept_id,
                    'name': concept_text,
                    'properties': {
                        'text': concept_text,
                        'source': 'pattern_extraction'
                    }
                })
        
        return concepts
    
    def extract_relationships(self, concepts: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Extract relationships between concepts (simple implementation)."""
        relationships = []
        
        # Simple co-occurrence based relationships
        concept_names = [c['name'] for c in concepts]
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                # Check if concepts appear near each other in text
                name1, name2 = concept1['name'], concept2['name']
                
                # Find positions of both concepts
                pos1 = text.lower().find(name1.lower())
                pos2 = text.lower().find(name2.lower())
                
                if pos1 != -1 and pos2 != -1 and abs(pos1 - pos2) < 100:
                    relationships.append({
                        'from_id': concept1['id'],
                        'to_id': concept2['id'],
                        'type': 'RELATED_TO',
                        'properties': {
                            'distance': abs(pos1 - pos2),
                            'source': 'co_occurrence'
                        }
                    })
        
        return relationships 