import logging
import asyncio
import json
import re
from typing import Dict, Any, Optional, List
from neo4j import AsyncGraphDatabase
from .abstract_agent import AbstractAgent
from config import config

logger = logging.getLogger(__name__)

class GraphRagBot(AbstractAgent):
    """
    Agentic Graph RAG Bot that uses Ollama with Neo4j for intelligent graph analysis.
    """
    
    def __init__(self, name: str = "Graph RAG Bot", description: str = "Agentic Graph RAG Bot with Neo4j Knowledge Graph", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, description, agent_config)
        
        # Neo4j connection
        self.neo4j_driver = None
        self.ollama_client = None
        self.initialized = False
        
        # Available tools for the agent
        self.available_tools = [
            {
                "name": "get_neo4j_schema",
                "description": "Get the current Neo4j graph schema including nodes, relationships, and properties",
                "parameters": []
            },
            {
                "name": "read_neo4j_cypher",
                "description": "Execute a read-only Cypher query on the Neo4j database",
                "parameters": ["query", "params (optional JSON string)"]
            },
            {
                "name": "write_neo4j_cypher", 
                "description": "Execute a write Cypher query on the Neo4j database",
                "parameters": ["query", "params (optional JSON string)"]
            }
        ]
        
        # System prompt for agentic behavior with Ollama
        self.system_prompt = """You are a Graph RAG expert that helps answer questions using a Neo4j knowledge graph.

WORKFLOW:
1. First, get the current Neo4j schema to understand the graph structure
2. Analyze the user's question and identify relevant nodes/relationships  
3. Formulate appropriate Cypher queries to find relevant information
4. Execute the queries using the available tools
5. Synthesize the results into a comprehensive answer

TOOL CALLING FORMAT:
To call a tool, use this exact format:
TOOL_CALL: <tool_name>
PARAMETERS: <parameter_values>
END_TOOL_CALL

Available tools:
- get_neo4j_schema: Get the current graph schema (no parameters needed)
- read_neo4j_cypher: Execute read-only Cypher queries (parameters: query, params)
- write_neo4j_cypher: Execute write Cypher queries (parameters: query, params)

Examples:
TOOL_CALL: get_neo4j_schema
END_TOOL_CALL

TOOL_CALL: read_neo4j_cypher
PARAMETERS: {"query": "MATCH (n:Company) RETURN n.name LIMIT 5"}
END_TOOL_CALL

TOOL_CALL: write_neo4j_cypher
PARAMETERS: {"query": "MERGE (c:Concept {name: $name})", "params": "{\\"name\\": \\"AI\\"}"}
END_TOOL_CALL

GUIDELINES:
- Always start by getting the schema to understand what data is available
- Write efficient Cypher queries that target the specific information needed
- Use LIMIT clauses to avoid returning too much data
- Explain your reasoning and show the query results
- Provide a clear, comprehensive answer based on the graph data
- When you execute queries, interpret the results and provide insights
- Be conversational and helpful in your explanations
"""
    
    async def initialize(self) -> bool:
        """Initialize the Graph RAG bot with Neo4j and Ollama."""
        self.logger.info("Initializing Agentic Graph RAG bot...")
        
        try:
            # Initialize Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            
            # Test connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            
            # Initialize Ollama client
            try:
                import ollama
                self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
                # Test Ollama connection
                self.ollama_client.list()
            except ImportError:
                raise ImportError("Ollama package not installed. Run: pip install ollama")
            except Exception as e:
                self.logger.error(f"Failed to connect to Ollama: {e}")
                return False
            
            # Load sample content if graph is empty
            node_count = await self._get_node_count()
            if node_count == 0:
                await self._load_sample_content()
            
            self.initialized = True
            self.logger.info(f"Agentic Graph RAG bot initialized successfully with {node_count} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Agentic Graph RAG bot: {e}")
            return False
    
    async def invoke(self, user_message: str) -> str:
        """Invoke the agentic Graph RAG bot to process queries."""
        self.logger.info(f"Agentic Graph RAG bot processing message: {user_message}")
        
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return "❌ Agentic Graph RAG bot initialization failed. Please check Neo4j connection and Ollama service."
        
        try:
            # Start conversation with system prompt
            conversation_history = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Run agentic loop with tool calls
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Create conversation prompt for Ollama
                prompt = self._format_conversation_for_ollama(conversation_history)
                
                # Call Ollama
                if not self.ollama_client:
                    return "❌ Ollama client not initialized"
                
                response = self.ollama_client.generate(
                    model=config.OLLAMA_MODEL,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "num_predict": 2000,
                    }
                )
                
                assistant_response = response['response'] if isinstance(response, dict) and 'response' in response else ''
                
                # Add assistant response to conversation
                conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                
                # Check for tool calls in the response
                tool_calls = self._extract_tool_calls(assistant_response)
                
                if not tool_calls:
                    # No more tool calls, return final response
                    return f"""🤖 **Agentic Graph RAG Response**

**Your Question:** {user_message}

**Analysis & Results:**
{assistant_response}

---
*This response was generated using agentic workflow with Neo4j graph analysis via Ollama*
"""
                
                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_params = tool_call.get("parameters", {})
                    
                    # Execute the tool
                    if tool_name:
                        result = await self._execute_tool(tool_name, tool_params)
                    else:
                        result = "Error: Tool name not specified"
                    
                    # Add tool result to conversation
                    conversation_history.append({
                        "role": "system",
                        "content": f"TOOL_RESULT for {tool_name}: {result}"
                    })
            
            # If we reach max iterations, return what we have
            return f"""🤖 **Agentic Graph RAG Response** (Max iterations reached)

**Your Question:** {user_message}

**Analysis & Results:**
The agentic analysis completed {max_iterations} iterations. Here's what was found:

{conversation_history[-2].get('content', 'Analysis in progress...')}

---
*This response was generated using agentic workflow with Neo4j graph analysis via Ollama*
"""
            
        except Exception as e:
            self.logger.error(f"Error in Agentic Graph RAG bot invoke: {e}")
            return f"❌ Error processing your query: {str(e)}"
    
    def _format_conversation_for_ollama(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation history for Ollama prompt."""
        prompt = ""
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Ollama response text."""
        tool_calls = []
        
        # Pattern to match TOOL_CALL blocks
        pattern = r'TOOL_CALL:\s*(\w+)\s*(?:PARAMETERS:\s*(.+?))?\s*END_TOOL_CALL'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            tool_name = match[0].strip()
            params_str = match[1].strip() if match[1] else "{}"
            
            # Parse parameters
            try:
                if params_str and params_str != "":
                    # Try to parse as JSON
                    parameters = json.loads(params_str)
                else:
                    parameters = {}
            except json.JSONDecodeError:
                # If not valid JSON, treat as empty parameters
                parameters = {}
            
            tool_calls.append({
                "name": tool_name,
                "parameters": parameters
            })
        
        return tool_calls
    
    async def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> str:
        """Execute a tool function and return the result."""
        try:
            if function_name == "get_neo4j_schema":
                return await self._get_schema()
            elif function_name == "read_neo4j_cypher":
                query = function_args.get("query", "")
                params = function_args.get("params", "{}")
                return await self._execute_read_query(query, params)
            elif function_name == "write_neo4j_cypher":
                query = function_args.get("query", "")
                params = function_args.get("params", "{}")
                return await self._execute_write_query(query, params)
            else:
                return f"Unknown function: {function_name}"
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    async def _get_schema(self) -> str:
        """Get the Neo4j schema."""
        try:
            async with self.neo4j_driver.session() as session:
                # Try APOC first, fallback to basic queries
                try:
                    result = await session.run("CALL apoc.meta.schema()")
                    record = await result.single()
                    if record:
                        schema = record['value']
                        return json.dumps(schema, default=str, indent=2)
                except:
                    pass
                
                # Basic schema query
                result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as node_labels")
                labels_record = await result.single()
                
                result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as relationship_types")
                rels_record = await result.single()
                
                schema = {
                    "node_labels": labels_record["node_labels"] if labels_record else [],
                    "relationship_types": rels_record["relationship_types"] if rels_record else []
                }
                
                return json.dumps(schema, indent=2)
        except Exception as e:
            return f"Error getting schema: {e}"
    
    async def _execute_read_query(self, query: str, params: str = "{}") -> str:
        """Execute a read-only Cypher query."""
        try:
            # Validate it's a read query
            if not self._is_read_query(query):
                return "Error: Only read queries (MATCH, RETURN, etc.) are allowed"
            
            parsed_params = json.loads(params) if params else {}
            
            async with self.neo4j_driver.session() as session:
                result = await session.run(query, parsed_params)
                records = await result.to_eager_result()
                
                # Convert to list of dictionaries
                data = [record.data() for record in records.records]
                
                return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return f"Error executing read query: {e}"
    
    async def _execute_write_query(self, query: str, params: str = "{}") -> str:
        """Execute a write Cypher query."""
        try:
            # Validate it's a write query
            if not self._is_write_query(query):
                return "Error: Only write queries (CREATE, MERGE, SET, DELETE, etc.) are allowed"
            
            parsed_params = json.loads(params) if params else {}
            
            async with self.neo4j_driver.session() as session:
                result = await session.run(query, parsed_params)
                summary = result.summary()
                
                counters = {
                    "nodes_created": summary.counters.nodes_created,
                    "nodes_deleted": summary.counters.nodes_deleted,
                    "relationships_created": summary.counters.relationships_created,
                    "relationships_deleted": summary.counters.relationships_deleted,
                    "properties_set": summary.counters.properties_set
                }
                
                return json.dumps(counters, indent=2)
        except Exception as e:
            return f"Error executing write query: {e}"
    
    def _is_read_query(self, query: str) -> bool:
        """Check if the query is a read-only query."""
        write_keywords = r'\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b'
        return not re.search(write_keywords, query, re.IGNORECASE)
    
    def _is_write_query(self, query: str) -> bool:
        """Check if the query is a write query."""
        write_keywords = r'\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b'
        return re.search(write_keywords, query, re.IGNORECASE) is not None
    
    async def _get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run("MATCH (n) RETURN count(n) as count")
                record = await result.single()
                return record["count"] if record else 0
        except Exception as e:
            self.logger.error(f"Error getting node count: {e}")
            return 0
    
    async def _load_sample_content(self):
        """Load sample content into the graph if it's empty."""
        try:
            async with self.neo4j_driver.session() as session:
                # Create main Healthee node
                await session.run("""
                    MERGE (h:Company {name: 'Healthee'})
                    SET h.description = $description,
                        h.type = 'health_benefits_platform',
                        h.updated_at = datetime()
                """, description="Technology-driven health benefits platform")
                
                # Create feature nodes
                features = [
                    "AI Assistant Zoe",
                    "Benefits Navigation", 
                    "Plan Comparison",
                    "Wellness Tools",
                    "Cost Savings",
                    "Real-time Support"
                ]
                
                for feature in features:
                    await session.run("""
                        MERGE (f:Feature {name: $feature})
                        SET f.category = 'platform_feature',
                            f.updated_at = datetime()
                        WITH f
                        MATCH (h:Company {name: 'Healthee'})
                        MERGE (h)-[:OFFERS]->(f)
                    """, feature=feature)
                
                # Create stakeholder nodes
                stakeholders = [
                    {"name": "Employees", "type": "user_group"},
                    {"name": "HR Teams", "type": "admin_group"},
                    {"name": "Organizations", "type": "client_group"}
                ]
                
                for stakeholder in stakeholders:
                    await session.run("""
                        MERGE (s:Stakeholder {name: $name})
                        SET s.type = $type,
                            s.updated_at = datetime()
                        WITH s
                        MATCH (h:Company {name: 'Healthee'})
                        MERGE (h)-[:SERVES]->(s)
                    """, name=stakeholder["name"], type=stakeholder["type"])
                
                # Create specific relationships
                await session.run("""
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
                
            self.logger.info("Sample content loaded into Neo4j graph")
                
        except Exception as e:
            self.logger.error(f"Error loading sample content: {e}")
    
    # Keep existing interface methods
    async def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add new knowledge to the graph using agentic workflow."""
        if not self.initialized:
            return False
        
        try:
            knowledge_prompt = f"""
            Please analyze the following content and add relevant nodes and relationships to the knowledge graph:

            Content: {content}
            Metadata: {json.dumps(metadata or {})}

            Steps:
            1. First get the current schema to understand existing structure
            2. Identify key entities, concepts, and relationships in the content
            3. Create appropriate Cypher queries to add this information to the graph
            4. Execute the queries to update the graph
            5. Confirm the additions were successful
            """
            
            result = await self.invoke(knowledge_prompt)
            self.logger.info(f"Added knowledge using agentic workflow")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def explore_concept(self, concept_name: str) -> Dict[str, Any]:
        """Explore a specific concept using agentic workflow."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            explore_prompt = f"""
            Please explore the concept '{concept_name}' in the Neo4j graph:
            
            1. First get the schema to understand the graph structure
            2. Search for nodes related to '{concept_name}'
            3. Find all relationships and connected concepts
            4. Provide a comprehensive analysis of this concept's role in the graph
            5. Include specific examples and connections found
            """
            
            result = await self.invoke(explore_prompt)
            
            return {
                "concept": concept_name,
                "analysis": result,
                "framework": "agentic_workflow"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """Get Agentic Graph RAG bot information."""
        info = super().get_info()
        info.update({
            "initialized": self.initialized,
            "database_type": "neo4j",
            "framework": "openai_function_calling",
            "capabilities": [
                "agentic_reasoning", 
                "graph_schema_analysis", 
                "cypher_query_generation",
                "multi_turn_reasoning",
                "knowledge_graph_management"
            ]
        })
        return info
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Agentic Graph RAG bot statistics."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            stats_prompt = """
            Please analyze the current Neo4j graph and provide comprehensive statistics including:
            1. Get the schema to understand node types and relationships
            2. Count nodes by type
            3. Count relationships by type
            4. Identify the most connected nodes
            5. Provide a summary of the graph structure
            
            Format the response as a clear summary of the graph statistics.
            """
            
            result = await self.invoke(stats_prompt)
            node_count = await self._get_node_count()
            
            return {
                "total_nodes": node_count,
                "graph_analysis": result,
                "framework": "openai_function_calling",
                "neo4j_healthy": await self.health_check()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if the Agentic Graph RAG bot is healthy."""
        if not self.initialized or not self.neo4j_driver:
            return False
        
        try:
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close Agentic Graph RAG bot resources."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        self.initialized = False
        self.logger.info("Agentic Graph RAG bot resources closed")
