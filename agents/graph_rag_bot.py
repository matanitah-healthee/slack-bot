from agents.abstract_agent import AbstractAgent
from neo4j import GraphDatabase
from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever, HybridRetriever
from neo4j_graphrag.generation import GraphRAG
from typing import List, Dict, Any, Optional
import os
import logging
import ollama
from dotenv import load_dotenv
from utils.embedding_utils import TextEmbedder

load_dotenv()

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

class CustomSentenceTransformerEmbeddings(Embedder):
    """Custom SentenceTransformer embeddings implementation for neo4j-graphrag compatibility.
    Uses the same embedding model as rag_bot for consistency."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.embedder = TextEmbedder(model_name)
        self.initialized = False
    
    def _ensure_initialized(self):
        """Ensure the embedder is initialized."""
        if not self.initialized:
            success = self.embedder.initialize()
            if not success:
                raise RuntimeError("Failed to initialize embedding model")
            self.initialized = True
    
    def embed_query(self, text: str) -> List[float]:
        """Synchronous embedding method."""
        try:
            self._ensure_initialized()
            return self.embedder.embed_text(text)
        except Exception as e:
            logger.error(f"SentenceTransformer embeddings error: {e}")
            return []
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embedding method."""
        return self.embed_query(text)

class GraphRagBot(AbstractAgent):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.initialized = False
        self.vector_index_name = "graph_rag_chunks"
        self.fulltext_index_name = None
        self.mcp_server = None
        

    async def initialize(self) -> bool:
        """Initialize the agent."""
        # Get environment variables with proper error handling
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER") 
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        
        # Validate required environment variables
        if not all([self.NEO4J_URI, self.NEO4J_USER, self.NEO4J_PASSWORD]):
            raise ValueError("Missing required Neo4j environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        
        # Initialize Neo4j driver (cast to str since we validated they're not None)
        neo4j_uri = str(self.NEO4J_URI)
        neo4j_user = str(self.NEO4J_USER)
        neo4j_password = str(self.NEO4J_PASSWORD)
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )

        # Get Neo4j schema
        schema_data = await self.get_neo4j_schema()
        self.neo4j_schema = str(schema_data) if schema_data else ""
        self.examples = [
            "USER INPUT: 'How many patients are over 50 years old?' QUERY: MATCH (p:Patient) WHERE p.patient_age > 50 RETURN count(p) as count",
            "USER INPUT: 'Which patients have submitted claims amounting to more than $1000?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE c.claim_amount > 1000 RETURN DISTINCT p.patient_id, c.claim_amount",
            "USER INPUT: 'What is the total claim amount for married patients?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_marital_status = 'Married' RETURN sum(c.claim_amount) as total",
            "USER INPUT: 'How many claims were submitted online in 2024?' QUERY: MATCH (c:Claim) WHERE c.claim_submission_method = 'Online' AND c.claim_date STARTS WITH '2024' RETURN count(c) as count",
            "USER INPUT: 'Which providers specialize in Cardiology?' QUERY: MATCH (p:Provider) WHERE p.provider_specialty = 'Cardiology' RETURN p.provider_id, p.provider_location",
            "USER INPUT: 'How many emergency claims were approved?' QUERY: MATCH (c:Claim) WHERE c.claim_type = 'Emergency' AND c.claim_status = 'Approved' RETURN count(c) as count",
            "USER INPUT: 'What are the most common diagnosis codes for patients under 30?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim)-[:includes_diagnosis]->(d:Diagnosis) WHERE p.patient_age < 30 RETURN d.diagnosis_code, count(*) as frequency ORDER BY frequency DESC",
            "USER INPUT: 'Which patients have both a diagnosis and a procedure in their claims?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim)-[:includes_diagnosis]->(d:Diagnosis), (c)-[:includes_procedure]->(pr:Procedure) RETURN DISTINCT p.patient_id",
            "USER INPUT: 'How many unemployed patients have claims pending?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_employment_status = 'Unemployed' AND c.claim_status = 'Pending' RETURN count(DISTINCT p) as count",
            "USER INPUT: 'What is the average claim amount for outpatient claims?' QUERY: MATCH (c:Claim) WHERE c.claim_type = 'Outpatient' RETURN avg(c.claim_amount) as average",
            "USER INPUT: 'Which providers have treated patients with an income over $100,000?' QUERY: MATCH (p:Patient)-[:has_provider]->(pr:Provider) WHERE p.patient_income > 100000 RETURN DISTINCT pr.provider_id, pr.provider_specialty",
            "USER INPUT: 'How many claims include a specific procedure code 'P123'?' QUERY: MATCH (c:Claim)-[:includes_procedure]->(p:Procedure) WHERE p.procedure_code = 'P123' RETURN count(c) as count",
            "USER INPUT: 'What is the gender distribution of patients with denied claims?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE c.claim_status = 'Denied' RETURN p.patient_gender, count(p) as count",
            "USER INPUT: 'Which diagnosis codes are associated with inpatient claims?' QUERY: MATCH (c:Claim)-[:includes_diagnosis]->(d:Diagnosis) WHERE c.claim_type = 'Inpatient' RETURN DISTINCT d.diagnosis_code",
            "USER INPUT: 'How many patients have multiple claims?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WITH p, count(c) as claim_count WHERE claim_count > 1 RETURN count(p) as count",
            "USER INPUT: 'What is the total claim amount by provider specialty?' QUERY: MATCH (p:Patient)-[:has_provider]->(pr:Provider), (p)-[:has_claim]->(c:Claim) RETURN pr.provider_specialty, sum(c.claim_amount) as total ORDER BY total DESC",
            "USER INPUT: 'Which patients over 65 have emergency claims?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_age > 65 AND c.claim_type = 'Emergency' RETURN p.patient_id, c.claim_id",
            "USER INPUT: 'How many claims were submitted by phone for retired patients?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_employment_status = 'Retired' AND c.claim_submission_method = 'Phone' RETURN count(c) as count",
            "USER INPUT: 'What are the top 5 procedure codes by frequency?' QUERY: MATCH (c:Claim)-[:includes_procedure]->(p:Procedure) RETURN p.procedure_code, count(*) as frequency ORDER BY frequency DESC LIMIT 5",
            "USER INPUT: 'Which providers in 'New York' have handled outpatient claims?' QUERY: MATCH (pr:Provider)<-[:has_provider]-(p:Patient)-[:has_claim]->(c:Claim) WHERE pr.provider_location = 'New York' AND c.claim_type = 'Outpatient' RETURN DISTINCT pr.provider_id",
            "USER INPUT: 'How many single patients have a claim amount over $500?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_marital_status = 'Single' AND c.claim_amount > 500 RETURN count(DISTINCT p) as count",
            "USER INPUT: 'What is the average age of patients with approved claims?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE c.claim_status = 'Approved' RETURN avg(p.patient_age) as average_age",
            "USER INPUT: 'Which diagnosis codes appear in claims from General Practice providers?' QUERY: MATCH (pr:Provider)<-[:has_provider]-(p:Patient)-[:has_claim]->(c:Claim)-[:includes_diagnosis]->(d:Diagnosis) WHERE pr.provider_specialty = 'General Practice' RETURN DISTINCT d.diagnosis_code",
            "USER INPUT: 'How many claims were filed in the last 6 months?' QUERY: MATCH (c:Claim) WHERE c.claim_date >= '2024-10-03' RETURN count(c) as count",
            "USER INPUT: 'Which patients have claims with both Cardiology and Orthopedics providers?' QUERY: MATCH (p:Patient)-[:has_provider]->(pr1:Provider {provider_specialty: 'Cardiology'}), (p)-[:has_provider]->(pr2:Provider {provider_specialty: 'Orthopedics'}) RETURN DISTINCT p.patient_id",
            "USER INPUT: 'What is the total claim amount for female patients under 40?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_gender = 'Female' AND p.patient_age < 40 RETURN sum(c.claim_amount) as total",
            "USER INPUT: 'How many procedures are linked to denied claims?' QUERY: MATCH (c:Claim)-[:includes_procedure]->(p:Procedure) WHERE c.claim_status = 'Denied' RETURN count(p) as count",
            "USER INPUT: 'Which patients have claims submitted on '2024-01-15'?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE c.claim_date = '2024-01-15' RETURN p.patient_id, c.claim_id",
            "USER INPUT: 'What is the distribution of claim types for widowed patients?' QUERY: MATCH (p:Patient)-[:has_claim]->(c:Claim) WHERE p.patient_marital_status = 'Widowed' RETURN c.claim_type, count(c) as count",
            "USER INPUT: 'How many students have claims with a Neurology provider?' QUERY: MATCH (p:Patient)-[:has_provider]->(pr:Provider), (p)-[:has_claim]->(c:Claim) WHERE p.patient_employment_status = 'Student' AND pr.provider_specialty = 'Neurology' RETURN count(DISTINCT p) as count"
        ]

        # Initialize embeddings
        self.embedder = CustomSentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        
        try:
            self.llm = CustomOllamaLLM(model_name="llama2")
            
            # Initialize retriever
            self.retriever = Text2CypherRetriever(
                driver=self.neo4j_driver,
                llm=self.llm,
                neo4j_schema=self.neo4j_schema,
                examples = self.examples
            )

            # Initialize GraphRAG
            self.graph_rag = GraphRAG(
                retriever=self.retriever,
                llm=self.llm
            )

        except Exception as e:
            self.logger.warning(f"Could not initialize Ollama LLM: {e}")
            self.llm = None
            self.graph_rag = None
        


        # Load sample content if database is empty
        node_count = await self._get_node_count()
        if node_count == 0:
            await self._load_sample_content()
        
        # Return True even if LLM initialization failed - the basic components are initialized
        self.initialized = True
        return True

    
    async def invoke(self, message: str):
        """Invoke the agent."""
        if self.graph_rag is None:
            return "GraphRAG not properly initialized. Please check Neo4j and Ollama configuration."
        
        try:
            response = self.graph_rag.search(query_text=message)
            return response.answer
        except Exception as e:
            self.logger.error(f"Error during GraphRAG search: {e}")
            return f"Error processing query: {str(e)}"

    async def _get_node_count(self) -> int:
        """Get total number of nodes in the Neo4j database."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                record = result.single()
                return record["node_count"] if record else 0
        except Exception as e:
            self.logger.error(f"Error getting node count: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check if Neo4j connection is healthy."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1")
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"Neo4j health check failed: {e}")
            return False

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

    async def get_neo4j_schema(self):
        """Get the Neo4j schema."""
        with self.neo4j_driver.session() as session:
            result = session.run("CALL apoc.meta.schema()")
            print(f"Neo4j schema: {result.data()}")
            return result.data()

    async def close(self):
        """Close Modern Graph RAG bot resources."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        self.initialized = False
        self.logger.info("Modern Graph RAG bot resources closed")