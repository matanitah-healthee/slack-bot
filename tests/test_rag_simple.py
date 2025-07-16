#!/usr/bin/env python3

import asyncio
import logging
import json
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.rag_bot import RagBot
from utils.vector_store import Document
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_system():
    """Test the RAG system with comprehensive debugging."""
    try:
        logger.info("Initializing RAG bot...")
        
        # Create RAG bot with default configuration
        rag_bot = RagBot(
            name="healthee_rag_test",
            description="Test RAG bot with sample content"
        )
        
        # Initialize the bot
        success = await rag_bot.initialize()
        if not success:
            logger.error("❌ Failed to initialize RAG bot")
            return False
        
        logger.info("✅ RAG bot initialized successfully")
        
        # Get initial stats
        stats = await rag_bot.get_stats()
        logger.info(f"Initial stats: {stats}")
        
        # Test queries with different similarity expectations
        test_queries = [
            "What is Healthee?",
            "Who does Healthee serve?", 
            "What are the key features?",
            "Tell me about Zoe",
            "healthee platform",
            "benefits navigation",
            "AI powered"
        ]
        
        logger.info("\n=== TESTING QUERIES ===")
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            try:
                response = await rag_bot.invoke(query)
                logger.info(f"Response length: {len(response)} characters")
                
                # Check if we got documents or fallback
                if "No relevant documents found" in response:
                    logger.warning(f"⚠️  No documents found for: {query}")
                else:
                    logger.info(f"✅ Found relevant documents for: {query}")
                    logger.info(f"Response preview: {response[:200]}...")
                
            except Exception as e:
                logger.error(f"❌ Error with query '{query}': {e}")
        
        # Test direct document search with very low threshold
        logger.info("\n=== TESTING DIRECT SEARCH ===")
        documents = await rag_bot.search_knowledge_base("Healthee", k=5)
        logger.info(f"Direct search found {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            logger.info(f"Document {i+1}: similarity={doc.get('similarity', 0):.3f}")
            logger.info(f"Content preview: {doc.get('content', '')[:100]}...")
        
        # Final stats
        final_stats = await rag_bot.get_stats()
        logger.info(f"\n=== FINAL STATS ===")
        for key, value in final_stats.items():
            logger.info(f"{key}: {value}")
        
        # Health check
        healthy = await rag_bot.health_check()
        logger.info(f"Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
        
        # Cleanup
        await rag_bot.close()
        
        logger.info("\n✅ RAG test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the RAG test."""
    logger.info("🚀 Starting RAG Bot Test")
    logger.info("=" * 50)
    
    success = await test_rag_system()
    
    if success:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.error("\n💥 Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 