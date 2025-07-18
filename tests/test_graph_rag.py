#!/usr/bin/env python3
"""
Healthcare Claims Graph RAG Bot Test

This script tests the Graph RAG bot with Neo4j knowledge graph operations
focusing on healthcare claims data including patient queries, provider analysis,
and healthcare relationship discovery using Text2Cypher capabilities.
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
        
        # Test 15 queries focusing on healthcare claims and patient data
        test_queries = [
            "How many patients are over 50 years old?",
            "Which patients have submitted claims amounting to more than $1000?",
            "What is the total claim amount for married patients?",
            "How many claims were submitted online in 2024?",
            "Which providers specialize in Cardiology?",
            "How many emergency claims were approved?",
            "What are the most common diagnosis codes for patients under 30?",
            "How many unemployed patients have claims pending?",
            "What is the average claim amount for outpatient claims?",
            "Which providers have treated patients with an income over $100,000?",
            "What is the gender distribution of patients with denied claims?",
            "How many patients have multiple claims?",
            "What is the total claim amount by provider specialty?",
            "Which patients over 65 have emergency claims?",
            "What are the top 5 procedure codes by frequency?"
        ]
        
        logger.info("\n=== TESTING CLAIMS QUERIES ===")
        
        for query in test_queries:
            logger.info(f"\nTesting claims query: '{query}'")
            
            try:
                response = await graph_bot.invoke(query)
                logger.info(f"Response length: {len(response)} characters")
                
                # Check if response looks like a Cypher query (expected output)
                if "MATCH" in response and "RETURN" in response:
                    logger.info("✅ Generated Cypher query")
                    logger.info(f"Query: {response}")
                elif "error" in response.lower():
                    logger.warning(f"⚠️  Error in response: {response}")
                else:
                    logger.info(f"🔍 Response: {response}")
                
            except Exception as e:
                logger.error(f"❌ Error with query '{query}': {e}")
        
        # Test specific healthcare data queries
        logger.info("\n=== TESTING SPECIFIC HEALTHCARE SCENARIOS ===")
        
        healthcare_scenarios = [
            "Show me all patients with high-value claims",
            "Find providers treating multiple patients", 
            "What claims are still pending approval?",
            "Which diagnosis codes appear most frequently?",
            "How many claims were denied last month?"
        ]
        
        for scenario in healthcare_scenarios:
            logger.info(f"\nTesting scenario: '{scenario}'")
            try:
                response = await graph_bot.invoke(scenario)
                logger.info(f"Response length: {len(response)} characters")
                
                if "MATCH" in response and "RETURN" in response:
                    logger.info("✅ Generated Cypher query for scenario")
                else:
                    logger.info(f"🔍 Scenario response: {response}")
                    
            except Exception as e:
                logger.error(f"❌ Error in scenario '{scenario}': {e}")
        
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
        
        # Test complex healthcare relationship queries  
        logger.info("\n=== TESTING HEALTHCARE RELATIONSHIP QUERIES ===")
        
        # Check if we can find relationships between healthcare entities
        relationship_queries = [
            "Find patients who have the same provider and similar diagnosis codes",
            "Show relationships between high-cost claims and provider specialties", 
            "Connect patients with multiple emergency claims to their providers",
            "Which diagnosis codes are most common for patients over 65?",
            "Find patterns between claim submission methods and approval rates"
        ]
        
        for relationship_query in relationship_queries:
            logger.info(f"\nHealthcare relationship query: '{relationship_query}'")
            try:
                response = await graph_bot.invoke(relationship_query)
                
                # Look for Cypher query indicators in response
                if "MATCH" in response and "RETURN" in response:
                    logger.info("✅ Generated relationship query")
                    logger.info(f"Cypher: {response}")
                elif "error" in response.lower():
                    logger.warning(f"⚠️  Error in relationship query: {response}")
                else:
                    logger.info(f"🔍 Relationship response: {response}")
                    
            except Exception as e:
                logger.error(f"❌ Error in relationship query: {e}")
        
        # Cleanup
        await graph_bot.close()
        
        logger.info("\n✅ Healthcare Claims Graph RAG test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Healthcare Claims Graph RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the Healthcare Claims Graph RAG test."""
    logger.info("🏥 Starting Healthcare Claims Graph RAG Bot Test")
    logger.info("=" * 60)
    
    # Check Neo4j configuration
    logger.info("Checking Neo4j configuration...")
    logger.info(f"Neo4j URI: {config.NEO4J_URI}")
    logger.info(f"Neo4j User: {config.NEO4J_USER}")
    logger.info("=" * 60)
    
    success = await test_graph_rag_system()
    
    if success:
        logger.info("\n🎉 All Healthcare Claims Graph RAG tests passed!")
    else:
        logger.error("\n💥 Some Healthcare Claims Graph RAG tests failed!")
        logger.info("\n🔧 Troubleshooting tips:")
        logger.info("1. Check if Neo4j is running and accessible")
        logger.info("2. Verify Neo4j credentials in config")
        logger.info("3. Ensure sample healthcare data is loaded")
        logger.info("4. Check if Ollama is running with llama2 model")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 