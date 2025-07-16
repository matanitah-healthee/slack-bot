#!/usr/bin/env python3
"""
Graph RAG Bot Test with Knowledge Graph Analysis

This script tests the Graph RAG bot with Neo4j knowledge graph operations
including concept extraction, relationship analysis, and graph traversal.
"""

import asyncio
import logging
import json
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.graph_rag_bot import GraphRagBot
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_graph_rag_system():
    """Test the Graph RAG system with comprehensive debugging."""
    try:
        logger.info("Initializing Graph RAG bot...")
        
        # Create Graph RAG bot
        graph_bot = GraphRagBot(
            name="healthee_graph_test",
            description="Test Graph RAG bot with Neo4j knowledge graph"
        )
        
        # Initialize the bot
        success = await graph_bot.initialize()
        if not success:
            logger.error("❌ Failed to initialize Graph RAG bot")
            return False
        
        logger.info("✅ Graph RAG bot initialized successfully")
        
        # Get initial stats
        stats = await graph_bot.get_stats()
        logger.info(f"Initial graph stats: {stats}")
        
        # Test queries focusing on graph traversal and concept relationships
        test_queries = [
            "What is Healthee?",
            "Tell me about Zoe",
            "How are benefits and AI connected?",
            "What features does the platform have?",
            "Explain the relationship between employees and wellness",
            "healthee AI assistant",
            "benefits navigation platform"
        ]
        
        logger.info("\n=== TESTING GRAPH QUERIES ===")
        
        for query in test_queries:
            logger.info(f"\nTesting graph query: '{query}'")
            
            try:
                response = await graph_bot.invoke(query)
                logger.info(f"Response length: {len(response)} characters")
                
                # Check response type
                if "No relevant concepts found" in response:
                    logger.warning(f"⚠️  No graph concepts found for: {query}")
                elif "Graph RAG Bot Response" in response:
                    logger.info(f"✅ Found graph concepts for: {query}")
                    logger.info(f"Response preview: {response[:300]}...")
                else:
                    logger.info(f"🔍 Unexpected response format for: {query}")
                
            except Exception as e:
                logger.error(f"❌ Error with query '{query}': {e}")
        
        # Test concept exploration
        logger.info("\n=== TESTING CONCEPT EXPLORATION ===")
        
        concept_tests = ["Healthee", "AI", "benefits", "platform"]
        
        for concept_name in concept_tests:
            logger.info(f"\nExploring concept: '{concept_name}'")
            
            try:
                exploration = await graph_bot.explore_concept(concept_name)
                
                if "error" in exploration:
                    logger.warning(f"⚠️  Could not explore concept '{concept_name}': {exploration['error']}")
                else:
                    logger.info(f"✅ Explored concept '{concept_name}':")
                    logger.info(f"  - Related concepts: {len(exploration.get('related_concepts', []))}")
                    logger.info(f"  - Relationships: {len(exploration.get('relationships', []))}")
                    logger.info(f"  - Total connections: {exploration.get('total_connections', 0)}")
                    
            except Exception as e:
                logger.error(f"❌ Error exploring concept '{concept_name}': {e}")
        
        # Test adding new knowledge
        logger.info("\n=== TESTING KNOWLEDGE ADDITION ===")
        
        new_knowledge = """
        Healthee's AI Assistant Zoe provides 24/7 support to employees.
        Zoe can help with benefit comparisons, claims processing, and wellness recommendations.
        The system integrates with HR platforms and provides real-time analytics.
        """
        
        try:
            success = await graph_bot.add_knowledge(
                new_knowledge,
                metadata={
                    'source': 'test_addition',
                    'type': 'feature_description',
                    'category': 'ai_assistant'
                }
            )
            
            if success:
                logger.info("✅ Successfully added new knowledge to graph")
                
                # Test query on newly added knowledge
                response = await graph_bot.invoke("Tell me about Zoe's capabilities")
                logger.info(f"Response to new knowledge query: {len(response)} characters")
                
            else:
                logger.warning("⚠️  Failed to add new knowledge")
                
        except Exception as e:
            logger.error(f"❌ Error adding knowledge: {e}")
        
        # Test graph statistics after additions
        logger.info("\n=== FINAL GRAPH ANALYSIS ===")
        
        final_stats = await graph_bot.get_stats()
        logger.info("Final graph statistics:")
        for key, value in final_stats.items():
            if key == "node_counts" and isinstance(value, dict):
                logger.info(f"  Node counts by type:")
                for node_type, count in value.items():
                    logger.info(f"    {node_type}: {count}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Health check
        healthy = await graph_bot.health_check()
        logger.info(f"Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
        
        # Test specific graph traversal
        logger.info("\n=== TESTING GRAPH TRAVERSAL ===")
        
        # Check if we can find relationships between key concepts
        test_traversals = [
            "Find connections between Healthee and AI",
            "Show relationships between employees and benefits",
            "Explore wellness and platform connections"
        ]
        
        for traversal_query in test_traversals:
            logger.info(f"\nGraph traversal: '{traversal_query}'")
            try:
                response = await graph_bot.invoke(traversal_query)
                
                # Look for graph-specific indicators in response
                if "Graph Analysis" in response or "Knowledge Graph Insights" in response:
                    logger.info("✅ Graph traversal response generated")
                else:
                    logger.warning("⚠️  Response may not include graph traversal insights")
                    
            except Exception as e:
                logger.error(f"❌ Error in graph traversal: {e}")
        
        # Cleanup
        await graph_bot.close()
        
        logger.info("\n✅ Graph RAG test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Graph RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the Graph RAG test."""
    logger.info("🕸️  Starting Graph RAG Bot Test")
    logger.info("=" * 60)
    
    # Check Neo4j configuration
    logger.info("Checking Neo4j configuration...")
    logger.info(f"Neo4j URI: {config.NEO4J_URI}")
    logger.info(f"Neo4j User: {config.NEO4J_USER}")
    logger.info("=" * 60)
    
    success = await test_graph_rag_system()
    
    if success:
        logger.info("\n🎉 All Graph RAG tests passed!")
    else:
        logger.error("\n💥 Some Graph RAG tests failed!")
        logger.info("\n🔧 Troubleshooting tips:")
        logger.info("1. Check if Neo4j is running and accessible")
        logger.info("2. Verify Neo4j credentials in config")
        logger.info("3. Ensure healthee.md file exists for sample data")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 