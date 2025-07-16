#!/usr/bin/env python3
"""
RAG System Setup Script

This script helps set up the RAG system with Ollama and PostgreSQL.
It will:
1. Check if all dependencies are installed
2. Check if Ollama is running
3. Check if PostgreSQL is available  
4. Initialize the RAG bot
5. Optionally scrape and index Healthee content
6. Test the system
"""

import asyncio
import os
import sys
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_modules = [
        ('asyncpg', 'PostgreSQL async driver'),
        ('ollama', 'Ollama client'),
        ('sentence_transformers', 'Text embeddings'),
        ('aiohttp', 'Web scraping'),
        ('bs4', 'HTML parsing'),
        ('numpy', 'Numerical operations')
    ]
    
    missing_modules = []
    
    for module_name, description in required_modules:
        try:
            __import__(module_name)
            logger.info(f"✅ {description} ({module_name}) is available")
        except ImportError:
            missing_modules.append((module_name, description))
            logger.error(f"❌ {description} ({module_name}) is missing")
    
    if missing_modules:
        logger.error("❌ Missing dependencies detected!")
        logger.info("Please install dependencies first:")
        logger.info("  uv pip install -r requirements.txt")
        logger.info("")
        logger.info("Missing modules:")
        for module_name, description in missing_modules:
            logger.info(f"  - {module_name}: {description}")
        return False
    
    logger.info("✅ All dependencies are available")
    return True

def check_ollama():
    """Check if Ollama is running."""
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        logger.info(f"✅ Ollama is running with {len(models.get('models', []))} models")
        
        # Check if llama2 model is available
        model_names = [model['name'] for model in models.get('models', [])]
        if 'llama2' in [name.split(':')[0] for name in model_names]:
            logger.info("✅ llama2 model is available")
        else:
            logger.warning("⚠️  llama2 model not found. You may need to run: ollama pull llama2")
            
        return True
    except Exception as e:
        logger.error(f"❌ Ollama check failed: {e}")
        logger.info("Please make sure Ollama is running: ollama serve")
        return False

async def check_postgres_async():
    """Async version of PostgreSQL check."""
    try:
        import asyncpg
    except ImportError:
        logger.error("❌ asyncpg module not found. Please install dependencies first:")
        logger.info("Run: uv pip install -r requirements.txt")
        return False
    
    try:
        conn = await asyncpg.connect("postgresql://postgres:password@localhost:5432/vectordb")
        
        # Check if pgvector extension is available
        result = await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        await conn.close()
        
        if result:
            logger.info("✅ PostgreSQL with pgvector is available")
            return True
        else:
            logger.error("❌ pgvector extension not found in database")
            return False
            
    except Exception as e:
        logger.error(f"❌ PostgreSQL check failed: {e}")
        logger.info("Please start PostgreSQL: docker-compose up -d")
        return False

def check_postgres():
    """Check if PostgreSQL with pgvector is available."""
    # This is a sync wrapper that will be called from async main
    return True  # We'll check it in main() instead

async def setup_rag_bot():
    """Set up and test the RAG bot."""
    try:
        from agents import RAGBot
        from config import config
        
        logger.info("Setting up RAG bot...")
        
        # Get agent configuration
        agent_config = config.get_agent_config()
        rag_config = agent_config['rag_bot']
        
        # Create RAG bot instance
        rag_bot = RAGBot(
            name="healthee_rag_test",
            description="Test RAG bot for Healthee knowledge",
            config=rag_config
        )
        
        # Initialize the bot
        success = await rag_bot.initialize()
        if not success:
            logger.error("❌ Failed to initialize RAG bot")
            return None
        
        logger.info("✅ RAG bot initialized successfully")
        return rag_bot
        
    except Exception as e:
        logger.error(f"❌ RAG bot setup failed: {e}")
        return None

async def scrape_and_index(rag_bot, max_pages: int = 10):
    """Scrape and index Healthee content."""
    try:
        logger.info(f"Scraping Healthee content (max {max_pages} pages)...")
        
        result = await rag_bot.scrape_and_index_healthee(max_pages=max_pages)
        
        if result.get('success'):
            logger.info(f"✅ Successfully indexed {result['chunks_created']} chunks from {result['pages_scraped']} pages")
        else:
            logger.error(f"❌ Scraping failed: {result.get('error', 'Unknown error')}")
            
        return result.get('success', False)
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return False


async def main():
    """Main setup function."""
    logger.info("🚀 Setting up RAG system with Ollama...")
    
    # Check prerequisites
    if not check_dependencies():
        return False
    
    if not check_ollama():
        return False
    
    if not await check_postgres_async():
        return False
    
    # Setup RAG bot
    rag_bot = await setup_rag_bot()
    if not rag_bot:
        return False
    
    # Check if we have existing data
    stats = await rag_bot.get_stats()
    doc_count = stats.get('document_count', 0)
    
    if doc_count == 0:
        logger.info("No documents found in vector store.")
        
        # Ask user if they want to scrape content
        response = input("Do you want to scrape and index Healthee content? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            max_pages = input("How many pages to scrape? (default: 10): ").strip()
            try:
                max_pages = int(max_pages) if max_pages else 10
            except ValueError:
                max_pages = 10
            
            success = await scrape_and_index(rag_bot, max_pages)
            if not success:
                logger.error("Failed to scrape content")
                return False
    else:
        logger.info(f"Found {doc_count} documents in vector store")
        
    logger.info("✅ RAG system setup complete!")
    logger.info("You can now run: python main.py")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 