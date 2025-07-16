"""
Graph Store utility for Neo4j database operations.
Handles knowledge graph creation, relationships, and traversal.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from neo4j import AsyncGraphDatabase
from config import config

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]

@dataclass
class GraphRelationship:
    """Represents a relationship between nodes."""
    from_node: str
    to_node: str
    relationship_type: str
    properties: Dict[str, Any]

class GraphStore:
    """Neo4j graph store for knowledge relationships."""
    
    def __init__(self):
        """Initialize the graph store."""
        self.driver = None
        
    async def initialize(self) -> bool:
        """Initialize the Neo4j connection."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            
            # Test connection
            await self.health_check()
            
            # Create indexes and constraints
            await self._create_schema()
            
            logger.info("Graph store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing graph store: {e}")
            return False
    
    async def _create_schema(self):
        """Create necessary indexes and constraints."""
        async with self.driver.session() as session:
            try:
                # Create constraint for unique document IDs
                await session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) 
                    REQUIRE d.id IS UNIQUE
                """)
                
                # Create constraint for unique concept IDs
                await session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) 
                    REQUIRE c.id IS UNIQUE
                """)
                
                # Create index on document content for text search
                await session.run("""
                    CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.content)
                """)
                
                # Create index on concept name
                await session.run("""
                    CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.name)
                """)
                
                logger.info("Graph schema created successfully")
                
            except Exception as e:
                logger.warning(f"Schema creation warning (may already exist): {e}")
    
    async def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a document node to the graph."""
        try:
            async with self.driver.session() as session:
                # Convert metadata to JSON string for Neo4j compatibility
                import json
                metadata_str = json.dumps(metadata) if metadata else "{}"
                
                await session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $content,
                        d.metadata_json = $metadata_str,
                        d.updated_at = datetime()
                """, doc_id=doc_id, content=content, metadata_str=metadata_str)
                
                logger.debug(f"Added document {doc_id} to graph")
                return True
                
        except Exception as e:
            logger.error(f"Error adding document to graph: {e}")
            return False
    
    async def add_concept(self, concept_id: str, name: str, properties: Dict[str, Any]) -> bool:
        """Add a concept node to the graph."""
        try:
            async with self.driver.session() as session:
                # Convert properties to JSON string for Neo4j compatibility
                import json
                properties_str = json.dumps(properties) if properties else "{}"
                
                await session.run("""
                    MERGE (c:Concept {id: $concept_id})
                    SET c.name = $name,
                        c.properties_json = $properties_str,
                        c.updated_at = datetime()
                """, concept_id=concept_id, name=name, properties_str=properties_str)
                
                logger.debug(f"Added concept {concept_id} to graph")
                return True
                
        except Exception as e:
            logger.error(f"Error adding concept to graph: {e}")
            return False
    
    async def create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two nodes."""
        try:
            properties = properties or {}
            
            async with self.driver.session() as session:
                await session.run(f"""
                    MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r += $properties,
                        r.updated_at = datetime()
                """, from_id=from_id, to_id=to_id, properties=properties)
                
                logger.debug(f"Created relationship {from_id} -[{rel_type}]-> {to_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    async def find_related_concepts(self, concept_id: str, max_depth: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """Find concepts related to a given concept through graph traversal."""
        try:
            async with self.driver.session() as session:
                result = await session.run("""
                    MATCH path = (start:Concept {id: $concept_id})-[*1..$max_depth]-(related)
                    WHERE related.id <> start.id
                    RETURN DISTINCT related.id as id, 
                           related.name as name,
                           related.properties as properties,
                           length(path) as distance
                    ORDER BY distance, related.name
                    LIMIT $limit
                """, concept_id=concept_id, max_depth=max_depth, limit=limit)
                
                related_concepts = []
                async for record in result:
                    concept = {
                        'id': record['id'],
                        'name': record['name'],
                        'properties': record['properties'],
                        'distance': record['distance']
                    }
                    related_concepts.append(concept)
                
                logger.debug(f"Found {len(related_concepts)} related concepts for {concept_id}")
                return related_concepts
                
        except Exception as e:
            logger.error(f"Error finding related concepts: {e}")
            return []
    
    async def search_concepts(self, search_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts by name or content."""
        try:
            async with self.driver.session() as session:
                result = await session.run("""
                    MATCH (c:Concept)
                    WHERE toLower(c.name) CONTAINS toLower($search_text)
                    RETURN c.id as id, c.name as name, c.properties_json as properties_json
                    ORDER BY c.name
                    LIMIT $limit
                """, search_text=search_query, limit=limit)
                
                concepts = []
                async for record in result:
                    import json
                    properties_json = record['properties_json'] or "{}"
                    properties = json.loads(properties_json) if properties_json else {}
                    
                    concept = {
                        'id': record['id'],
                        'name': record['name'],
                        'properties': properties
                    }
                    concepts.append(concept)
                
                logger.debug(f"Found {len(concepts)} concepts matching '{search_query}'")
                return concepts
                
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    async def get_concept_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a concept."""
        try:
            async with self.driver.session() as session:
                result = await session.run("""
                    MATCH (c:Concept {id: $concept_id})-[r]-(related)
                    RETURN type(r) as relationship_type,
                           related.id as related_id,
                           related.name as related_name,
                           r.properties as properties,
                           startNode(r).id = $concept_id as outgoing
                    ORDER BY relationship_type, related_name
                """, concept_id=concept_id)
                
                relationships = []
                async for record in result:
                    rel = {
                        'type': record['relationship_type'],
                        'related_id': record['related_id'],
                        'related_name': record['related_name'],
                        'properties': record['properties'],
                        'direction': 'outgoing' if record['outgoing'] else 'incoming'
                    }
                    relationships.append(rel)
                
                logger.debug(f"Found {len(relationships)} relationships for {concept_id}")
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting concept relationships: {e}")
            return []
    
    async def get_node_count(self) -> Dict[str, int]:
        """Get count of nodes by type."""
        try:
            async with self.driver.session() as session:
                result = await session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                    ORDER BY count DESC
                """)
                
                counts = {}
                async for record in result:
                    if record['label']:
                        counts[record['label']] = record['count']
                
                return counts
                
        except Exception as e:
            logger.error(f"Error getting node count: {e}")
            return {}
    
    async def get_relationship_count(self) -> int:
        """Get total count of relationships."""
        try:
            async with self.driver.session() as session:
                result = await session.run("MATCH ()-[r]-() RETURN count(r) as count")
                record = await result.single()
                return record['count'] if record else 0
                
        except Exception as e:
            logger.error(f"Error getting relationship count: {e}")
            return 0
    
    async def delete_all(self) -> bool:
        """Delete all nodes and relationships."""
        try:
            async with self.driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            
            logger.info("Deleted all nodes and relationships from graph store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting all data: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if the graph store is healthy."""
        try:
            if not self.driver:
                return False
            
            async with self.driver.session() as session:
                await session.run("RETURN 1")
                return True
                
        except Exception as e:
            logger.error(f"Graph store health check failed: {e}")
            return False
    
    async def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Graph store connection closed") 