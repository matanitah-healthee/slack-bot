import logging
import asyncio
import json
import ollama
from typing import Dict, Any, Optional, List
from fastmcp import FastMCP, Context
from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever, HybridRetriever
from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index
from .abstract_agent import AbstractAgent
from config import config

logger = logging.getLogger(__name__)

class CustomOllamaLLM(LLMInterface):
    """Custom Ollama LLM implementation for neo4j-graphrag compatibility."""
    
    def __init__(self, model_name: str, host: str = "http://localhost:11434", **kwargs):
        self.model_name = model_name
        self.host = host
        self.kwargs = kwargs
    
    def invoke(self, input_text: str) -> LLMResponse:
        """Synchronous invoke method."""
        try:
            # Convert kwargs to options format for ollama.chat
            options = {}
            if 'temperature' in self.kwargs:
                options['temperature'] = self.kwargs['temperature']
            if 'num_predict' in self.kwargs:
                options['num_predict'] = self.kwargs['num_predict']
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": input_text}],
                options=options if options else None
            )
            return LLMResponse(content=response["message"]["content"])
        except Exception as e:
            logger.error(f"Ollama LLM error: {e}")
            return LLMResponse(content="Error generating response")
    
    async def ainvoke(self, input_text: str) -> LLMResponse:
        """Async invoke method - just calls sync version."""
        return self.invoke(input_text)

class CustomOllamaEmbeddings(Embedder):
    """Custom Ollama embeddings implementation for neo4j-graphrag compatibility."""
    
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def embed_query(self, text: str) -> List[float]:
        """Synchronous embedding method."""
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Ollama embeddings error: {e}")
            return []
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embedding method."""
        return self.embed_query(text)

class GraphRagBot(AbstractAgent):
    """
    Modern Graph RAG Bot using neo4j-graphrag and fastmcp libraries.
    Provides intelligent graph analysis through standardized MCP tools.
    """
    
    def __init__(self, name: str = "Modern Graph RAG Bot", description: str = "Advanced Graph RAG with neo4j-graphrag and fastmcp", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, description, agent_config)
        
        # Core components
        self.neo4j_driver = None
        self.llm = None
        self.embedder = None
        self.graph_rag = None
        self.vector_retriever = None
        self.text2cypher_retriever = None
        self.hybrid_retriever = None
        self.mcp_server = None
        self.initialized = False
        
        # Vector index configuration
        self.vector_index_name = "graph_rag_chunks"
        self.fulltext_index_name = "graph_rag_fulltext"
        
    async def initialize(self) -> bool:
        """Initialize the Modern Graph RAG bot with neo4j-graphrag and fastmcp."""
        self.logger.info("Initializing Modern Graph RAG bot...")
        
        try:
            # Initialize Neo4j connection (synchronous driver for neo4j-graphrag)
            self.neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            
            # Test Neo4j connection
            await self._test_neo4j_connection()
            
            # Initialize LLM with custom Ollama implementation
            self.llm = CustomOllamaLLM(
                model_name=config.OLLAMA_MODEL,
                host=config.OLLAMA_BASE_URL,
                temperature=0.7,
                num_predict=2000,
            )
            
            # Initialize embedder with custom implementation (use dedicated embedding model)
            self.embedder = CustomOllamaEmbeddings(
                model="nomic-embed-text",  # Use dedicated embedding model
                host=config.OLLAMA_BASE_URL
            )
            
            # Setup vector indexes if they don't exist
            await self._setup_indexes()
            
            # Initialize retrievers
            await self._initialize_retrievers()
            
            # Initialize FastMCP server
            await self._initialize_mcp_server()
            
            # Load sample content if graph is empty
            node_count = await self._get_node_count()
            if node_count == 0:
                await self._load_sample_content()
            
            self.initialized = True
            self.logger.info(f"Modern Graph RAG bot initialized successfully with {node_count} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Modern Graph RAG bot: {e}")
            return False
    
    async def _test_neo4j_connection(self):
        """Test Neo4j connection."""
        if not self.neo4j_driver:
            raise RuntimeError("Neo4j driver not initialized")
        with self.neo4j_driver.session() as session:
            session.run("RETURN 1")
    
    async def _setup_indexes(self):
        """Setup necessary vector and fulltext indexes."""
        try:
            if not self.neo4j_driver:
                raise RuntimeError("Neo4j driver not initialized")
                
            # Create vector index for embeddings
            create_vector_index(
                self.neo4j_driver,
                self.vector_index_name,
                label="Chunk",
                embedding_property="embedding",
                dimensions=768,  # nomic-embed-text embedding dimension
                similarity_fn="cosine"
            )
            
            # Create fulltext index for text search
            create_fulltext_index(
                self.neo4j_driver,
                self.fulltext_index_name,
                label="Chunk",
                node_properties=["text"]
            )
            
            self.logger.info("Vector and fulltext indexes created successfully")
            
        except Exception as e:
            # Indexes might already exist, which is fine
            self.logger.debug(f"Index creation note: {e}")
    
    async def _initialize_retrievers(self):
        """Initialize different types of retrievers."""
        if not self.neo4j_driver or not self.llm or not self.embedder:
            raise RuntimeError("Required components not initialized")
            
        # Vector retriever for semantic search
        self.vector_retriever = VectorRetriever(
            driver=self.neo4j_driver,
            index_name=self.vector_index_name,
            embedder=self.embedder,
            return_properties=["text", "source"]
        )
        
        # Text2Cypher retriever for natural language to Cypher
        schema = await self._get_neo4j_schema()
        self.text2cypher_retriever = Text2CypherRetriever(
            driver=self.neo4j_driver,
            llm=self.llm,
            neo4j_schema=schema
        )
        
        # Hybrid retriever combining vector and fulltext search
        self.hybrid_retriever = HybridRetriever(
            driver=self.neo4j_driver,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            embedder=self.embedder
        )
        
        # Initialize main GraphRAG with vector retriever as default
        self.graph_rag = GraphRAG(
            retriever=self.vector_retriever,
            llm=self.llm
        )
    
    async def _initialize_mcp_server(self):
        """Initialize FastMCP server with graph analysis tools."""
        self.mcp_server = FastMCP(
            name="GraphRAG Analysis"
        )
        
        # Register MCP tools
        self._register_mcp_tools()
    
    def _register_mcp_tools(self):
        """Register MCP tools for graph analysis."""
        if not self.mcp_server:
            return
            
        @self.mcp_server.tool()
        async def semantic_search(query: str, top_k: int = 5, ctx: Optional[Context] = None) -> dict:
            """Perform semantic search using vector embeddings."""
            if ctx:
                await ctx.info(f"Performing semantic search for: {query}")
            
            try:
                result = self.graph_rag.search(
                    query_text=query,
                    retriever_config={"top_k": top_k},
                    return_context=True
                )
                
                if ctx:
                    await ctx.info("Semantic search completed successfully")
                return {
                    "answer": result.answer,
                    "context": [item.content for item in result.retriever_result.items] if result.retriever_result else [],
                    "method": "semantic_search"
                }
            except Exception as e:
                if ctx:
                    await ctx.error(f"Semantic search failed: {str(e)}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def cypher_query(question: str, ctx: Optional[Context] = None) -> dict:
            """Convert natural language to Cypher query and execute it."""
            if ctx:
                await ctx.info(f"Converting question to Cypher: {question}")
            
            try:
                # Use Text2Cypher retriever
                result = self.text2cypher_retriever.search(query_text=question)
                
                # Also get LLM interpretation
                rag_result = GraphRAG(
                    retriever=self.text2cypher_retriever,
                    llm=self.llm
                ).search(query_text=question, return_context=True)
                
                if ctx:
                    await ctx.info("Cypher query executed successfully")
                return {
                    "answer": rag_result.answer,
                    "raw_results": [item.content for item in result.items] if result.items else [],
                    "method": "cypher_query"
                }
            except Exception as e:
                if ctx:
                    await ctx.error(f"Cypher query failed: {str(e)}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5, ctx: Optional[Context] = None) -> dict:
            """Perform hybrid search combining vector and fulltext search."""
            if ctx:
                await ctx.info(f"Performing hybrid search for: {query}")
            
            try:
                rag_result = GraphRAG(
                    retriever=self.hybrid_retriever,
                    llm=self.llm
                ).search(
                    query_text=query,
                    retriever_config={"top_k": top_k, "alpha": alpha},
                    return_context=True
                )
                
                if ctx:
                    await ctx.info("Hybrid search completed successfully")
                return {
                    "answer": rag_result.answer,
                    "context": [item.content for item in rag_result.retriever_result.items] if rag_result.retriever_result else [],
                    "method": "hybrid_search"
                }
            except Exception as e:
                if ctx:
                    await ctx.error(f"Hybrid search failed: {str(e)}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def get_graph_schema(ctx: Optional[Context] = None) -> dict:
            """Get the current Neo4j graph schema."""
            if ctx:
                await ctx.info("Retrieving graph schema")
            
            try:
                schema = await self._get_neo4j_schema()
                if ctx:
                    await ctx.info("Schema retrieved successfully")
                return {"schema": schema, "method": "get_schema"}
            except Exception as e:
                if ctx:
                    await ctx.error(f"Schema retrieval failed: {str(e)}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def analyze_concept(concept: str, method: str = "semantic", ctx: Optional[Context] = None) -> dict:
            """Analyze a specific concept using the specified method."""
            if ctx:
                await ctx.info(f"Analyzing concept '{concept}' using {method} method")
            
            try:
                if method == "semantic":
                    retriever = self.vector_retriever
                elif method == "cypher":
                    retriever = self.text2cypher_retriever
                elif method == "hybrid":
                    retriever = self.hybrid_retriever
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                rag_result = GraphRAG(
                    retriever=retriever,
                    llm=self.llm
                ).search(
                    query_text=f"Tell me everything about {concept} and its relationships",
                    retriever_config={"top_k": 10},
                    return_context=True
                )
                
                if ctx:
                    await ctx.info(f"Concept analysis completed using {method}")
                return {
                    "concept": concept,
                    "analysis": rag_result.answer,
                    "context": [item.content for item in rag_result.retriever_result.items] if rag_result.retriever_result else [],
                    "method": method
                }
            except Exception as e:
                if ctx:
                    await ctx.error(f"Concept analysis failed: {str(e)}")
                return {"error": str(e)}
    
    async def invoke(self, user_message: str) -> str:
        """Main entry point for processing user queries."""
        self.logger.info(f"Modern Graph RAG bot processing message: {user_message}")
        
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return "❌ Modern Graph RAG bot initialization failed. Please check Neo4j connection and Ollama service."
        
        try:
            # Use the main GraphRAG for general queries (synchronous call)
            result = self.graph_rag.search(
                query_text=user_message,
                retriever_config={"top_k": 5},
                return_context=True
            )
            
            # Format response
            context_items = [item.content for item in result.retriever_result.items] if result.retriever_result else []
            
            return f"""🤖 **Modern Graph RAG Response**

**Your Question:** {user_message}

**Answer:** {result.answer}

**Sources:** Found {len(context_items)} relevant context items

---
*Powered by neo4j-graphrag and fastmcp*
"""
            
        except Exception as e:
            self.logger.error(f"Error in Modern Graph RAG bot invoke: {e}")
            return f"❌ Error processing your query: {str(e)}"
    
    async def _get_neo4j_schema(self) -> str:
        """Get Neo4j schema for Text2Cypher retriever."""
        try:
            with self.neo4j_driver.session() as session:
                # Get node labels
                result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                labels_record = result.single()
                labels = labels_record["labels"] if labels_record else []
                
                # Get relationship types
                result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
                types_record = result.single()
                rel_types = types_record["types"] if types_record else []
                
                # Build schema string
                schema = f"""
Node Labels: {', '.join(labels)}
Relationship Types: {', '.join(rel_types)}

Sample patterns:
(:Company)-[:OFFERS]->(:Feature)
(:Company)-[:SERVES]->(:Stakeholder)
(:Feature)-[:ASSISTS]->(:Stakeholder)
"""
                return schema
        except Exception as e:
            self.logger.error(f"Error getting schema: {e}")
            return "Schema unavailable"
    
    async def _get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                record = result.single()
                return record["count"] if record else 0
        except Exception as e:
            self.logger.error(f"Error getting node count: {e}")
            return 0
    
    async def _load_sample_content(self):
        """Load sample content with embeddings for demonstration."""
        try:
            with self.neo4j_driver.session() as session:
                # Create main company node
                session.run("""
                    MERGE (h:Company {name: 'Healthee'})
                    SET h.description = $description,
                        h.type = 'health_benefits_platform',
                        h.updated_at = datetime()
                """, description="Technology-driven health benefits platform")
                
                # Create feature nodes with sample text chunks
                features_data = [
                    {"name": "AI Assistant Zoe", "text": "Zoe is an AI-powered assistant that helps employees navigate their health benefits with personalized guidance and real-time support."},
                    {"name": "Benefits Navigation", "text": "Comprehensive benefits navigation system that simplifies complex health insurance plans and helps users find the right coverage."},
                    {"name": "Plan Comparison", "text": "Advanced plan comparison tools that analyze costs, coverage, and benefits to help users make informed decisions."},
                    {"name": "Wellness Tools", "text": "Integrated wellness tools including health tracking, preventive care reminders, and wellness program management."},
                    {"name": "Cost Savings", "text": "Intelligent cost optimization features that identify savings opportunities and help reduce healthcare expenses."},
                    {"name": "Real-time Support", "text": "24/7 real-time support system providing instant answers to benefits questions and claim assistance."}
                ]
                
                for feature_data in features_data:
                    # Create feature node
                    session.run("""
                        MERGE (f:Feature {name: $name})
                        SET f.category = 'platform_feature',
                            f.updated_at = datetime()
                        WITH f
                        MATCH (h:Company {name: 'Healthee'})
                        MERGE (h)-[:OFFERS]->(f)
                    """, name=feature_data["name"])
                    
                    # Create chunk with embedding
                    try:
                        embedding = self.embedder.embed_query(feature_data["text"])
                        session.run("""
                            MERGE (c:Chunk {id: $chunk_id})
                            SET c.text = $text,
                                c.source = $source,
                                c.embedding = $embedding,
                                c.updated_at = datetime()
                            WITH c
                            MATCH (f:Feature {name: $feature_name})
                            MERGE (f)-[:HAS_CHUNK]->(c)
                        """, 
                        chunk_id=f"chunk_{feature_data['name'].lower().replace(' ', '_')}",
                        text=feature_data["text"],
                        source=f"Feature: {feature_data['name']}",
                        embedding=embedding,
                        feature_name=feature_data["name"])
                    except Exception as e:
                        self.logger.warning(f"Failed to create embedding for {feature_data['name']}: {e}")
                
                # Create stakeholder nodes
                stakeholders = [
                    {"name": "Employees", "type": "user_group", "text": "End users who utilize the platform to manage their health benefits and make informed healthcare decisions."},
                    {"name": "HR Teams", "type": "admin_group", "text": "Human resources professionals who administer benefits programs and support employee wellness initiatives."},
                    {"name": "Organizations", "type": "client_group", "text": "Companies and organizations that partner with Healthee to provide comprehensive benefits solutions to their workforce."}
                ]
                
                for stakeholder in stakeholders:
                    session.run("""
                        MERGE (s:Stakeholder {name: $name})
                        SET s.type = $type,
                            s.updated_at = datetime()
                        WITH s
                        MATCH (h:Company {name: 'Healthee'})
                        MERGE (h)-[:SERVES]->(s)
                    """, name=stakeholder["name"], type=stakeholder["type"])
                    
                    # Create chunk for stakeholder
                    try:
                        embedding = self.embedder.embed_query(stakeholder["text"])
                        session.run("""
                            MERGE (c:Chunk {id: $chunk_id})
                            SET c.text = $text,
                                c.source = $source,
                                c.embedding = $embedding,
                                c.updated_at = datetime()
                            WITH c
                            MATCH (s:Stakeholder {name: $stakeholder_name})
                            MERGE (s)-[:HAS_CHUNK]->(c)
                        """, 
                        chunk_id=f"chunk_{stakeholder['name'].lower().replace(' ', '_')}",
                        text=stakeholder["text"],
                        source=f"Stakeholder: {stakeholder['name']}",
                        embedding=embedding,
                        stakeholder_name=stakeholder["name"])
                    except Exception as e:
                        self.logger.warning(f"Failed to create embedding for {stakeholder['name']}: {e}")
                
                # Create specific relationships
                session.run("""
                    MATCH (zoe:Feature {name: 'AI Assistant Zoe'})
                    MATCH (emp:Stakeholder {name: 'Employees'})
                    MERGE (zoe)-[:ASSISTS]->(emp)
                    
                    MATCH (zoe:Feature {name: 'AI Assistant Zoe'})
                    MATCH (support:Feature {name: 'Real-time Support'})
                    MERGE (zoe)-[:PROVIDES]->(support)
                    
                    MATCH (nav:Feature {name: 'Benefits Navigation'})
                    MATCH (comp:Feature {name: 'Plan Comparison'})
                    MERGE (nav)-[:INCLUDES]->(comp)
                """)
                
            self.logger.info("Sample content with embeddings loaded into Neo4j graph")
                
        except Exception as e:
            self.logger.error(f"Error loading sample content: {e}")
    
    # Legacy interface methods for compatibility
    async def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add new knowledge to the graph using modern GraphRAG components."""
        if not self.initialized:
            return False
        
        try:
            # Create embedding for the content
            embedding = self.embedder.embed_query(content)
            
            # Create chunk in graph
            chunk_id = f"chunk_{hash(content) % 10000}"
            with self.neo4j_driver.session() as session:
                session.run("""
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $content,
                        c.source = $source,
                        c.embedding = $embedding,
                        c.metadata = $metadata,
                        c.updated_at = datetime()
                """, 
                chunk_id=chunk_id,
                content=content,
                source=metadata.get("source", "Manual input") if metadata else "Manual input",
                embedding=embedding,
                metadata=json.dumps(metadata or {}))
            
            self.logger.info(f"Added knowledge chunk: {chunk_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def explore_concept(self, concept_name: str) -> Dict[str, Any]:
        """Explore a concept using modern GraphRAG retrievers."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            # Use semantic search (synchronous call)
            result = self.graph_rag.search(
                query_text=f"Tell me about {concept_name} and its relationships",
                retriever_config={"top_k": 5},
                return_context=True
            )
            
            return {
                "concept": concept_name,
                "analysis": result.answer,
                "context_items": len(result.retriever_result.items) if result.retriever_result else 0,
                "framework": "neo4j-graphrag"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """Get Modern Graph RAG bot information."""
        info = super().get_info()
        info.update({
            "initialized": self.initialized,
            "database_type": "neo4j",
            "framework": "neo4j-graphrag + fastmcp",
            "capabilities": [
                "semantic_search", 
                "cypher_queries", 
                "hybrid_search",
                "mcp_tools",
                "modern_retrievers",
                "vector_embeddings"
            ],
            "mcp_tools": [
                "semantic_search",
                "cypher_query", 
                "hybrid_search",
                "get_graph_schema",
                "analyze_concept"
            ]
        })
        return info
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Modern Graph RAG bot statistics."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            node_count = await self._get_node_count()
            
            # Get chunk count
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (c:Chunk) RETURN count(c) as chunk_count")
                record = result.single()
                chunk_count = record["chunk_count"] if record else 0
            
            return {
                "total_nodes": node_count,
                "chunk_nodes": chunk_count,
                "vector_index": self.vector_index_name,
                "fulltext_index": self.fulltext_index_name,
                "framework": "neo4j-graphrag + fastmcp",
                "neo4j_healthy": await self.health_check(),
                "mcp_server_active": self.mcp_server is not None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if the Modern Graph RAG bot is healthy."""
        if not self.initialized or not self.neo4j_driver:
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    def get_mcp_server(self) -> Optional[FastMCP]:
        """Get the FastMCP server instance for external usage."""
        return self.mcp_server
    
    async def close(self):
        """Close Modern Graph RAG bot resources."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        self.initialized = False
        self.logger.info("Modern Graph RAG bot resources closed")
