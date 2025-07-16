#!/usr/bin/env python3
"""
Database Setup Script for RAG and Graph RAG Bots

This script helps set up PostgreSQL with pgvector and Neo4j databases
for the agent system.
"""

import asyncio
import subprocess
import sys
import os
from typing import Dict, Any

async def check_postgres_connection():
    """Check if PostgreSQL with pgvector is available."""
    try:
        from utils.vector_store import VectorStore
        
        vector_store = VectorStore("setup_test")
        success = await vector_store.initialize()
        
        if success:
            await vector_store.close()
            return True, "PostgreSQL with pgvector is running and accessible"
        else:
            return False, "PostgreSQL connection failed"
            
    except Exception as e:
        return False, f"PostgreSQL error: {str(e)}"

async def check_neo4j_connection():
    """Check if Neo4j is available."""
    try:
        from utils.graph_store import GraphStore
        
        graph_store = GraphStore()
        success = await graph_store.initialize()
        
        if success:
            await graph_store.close()
            return True, "Neo4j is running and accessible"
        else:
            return False, "Neo4j connection failed"
            
    except Exception as e:
        return False, f"Neo4j error: {str(e)}"

def setup_postgresql_docker():
    """Set up PostgreSQL with pgvector using Docker."""
    print("🐳 Setting up PostgreSQL with pgvector using Docker...")
    
    try:
        # Check if docker-compose.yml exists
        if not os.path.exists('docker-compose.yml'):
            print("❌ docker-compose.yml not found. Creating one...")
            create_docker_compose()
        
        # Start PostgreSQL with pgvector
        result = subprocess.run(['docker-compose', 'up', '-d'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL with pgvector started successfully")
            print("   Connection: postgresql://postgres:password@localhost:5432/vectordb")
            return True
        else:
            print(f"❌ Failed to start PostgreSQL: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Docker or docker-compose not found. Please install Docker first.")
        return False
    except Exception as e:
        print(f"❌ Error setting up PostgreSQL: {e}")
        return False

def setup_neo4j_docker():
    """Set up Neo4j using Docker."""
    print("🐳 Setting up Neo4j using Docker...")
    
    try:
        # Run Neo4j container
        cmd = [
            'docker', 'run', '-d',
            '--name', 'neo4j-rag',
            '-p', '7474:7474',
            '-p', '7687:7687',
            '-e', 'NEO4J_AUTH=neo4j/password',
            '-e', 'NEO4J_apoc_export_file_enabled=true',
            '-e', 'NEO4J_apoc_import_file_enabled=true',
            '--restart', 'unless-stopped',
            'neo4j:5.16'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Neo4j started successfully")
            print("   Web UI: http://localhost:7474")
            print("   Connection: bolt://localhost:7687")
            print("   Credentials: neo4j/password")
            print("   ⏳ Waiting for Neo4j to fully start (this may take a minute)...")
            return True
        else:
            # Check if container already exists
            if "already in use by container" in result.stderr:
                print("✅ Neo4j container already exists")
                return True
            else:
                print(f"❌ Failed to start Neo4j: {result.stderr}")
                return False
            
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        return False
    except Exception as e:
        print(f"❌ Error setting up Neo4j: {e}")
        return False

def create_docker_compose():
    """Create a docker-compose.yml file for PostgreSQL with pgvector."""
    docker_compose_content = """version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: postgres-pgvector
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: vectordb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-pgvector.sql:/docker-entrypoint-initdb.d/init-pgvector.sql
    restart: unless-stopped

volumes:
  postgres_data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Created docker-compose.yml")

def install_dependencies():
    """Install required Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

async def test_agent_initialization():
    """Test that agents can be initialized with the databases."""
    print("🧪 Testing agent initialization...")
    
    try:
        from agents.agent_manager import agent_manager
        
        # Get agents
        agents = agent_manager.list_agents()
        
        for agent_info in agents:
            agent = agent_manager.get_agent(agent_info['id'])
            if agent:
                try:
                    success = await agent.initialize()
                    if success:
                        print(f"✅ {agent_info['name']} initialized successfully")
                        
                        # Get stats if available
                        if hasattr(agent, 'get_stats') and callable(getattr(agent, 'get_stats')):
                            stats = await agent.get_stats()
                            print(f"   Stats: {stats}")
                        
                        # Close the agent
                        if hasattr(agent, 'close'):
                            await agent.close()
                    else:
                        print(f"❌ {agent_info['name']} initialization failed")
                except Exception as e:
                    print(f"❌ {agent_info['name']} error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
        return False

async def main():
    """Main setup function."""
    print("🚀 RAG and Graph RAG Bot Database Setup")
    print("=" * 50)
    
    setup_steps = []
    
    # Check current status
    print("📋 Checking current database status...")
    
    postgres_ok, postgres_msg = await check_postgres_connection()
    print(f"PostgreSQL: {'✅' if postgres_ok else '❌'} {postgres_msg}")
    
    neo4j_ok, neo4j_msg = await check_neo4j_connection()
    print(f"Neo4j: {'✅' if neo4j_ok else '❌'} {neo4j_msg}")
    
    if postgres_ok and neo4j_ok:
        print("\n🎉 All databases are already running!")
        test_success = await test_agent_initialization()
        return test_success
    
    print("\n🔧 Setting up missing databases...")
    
    # Install dependencies first
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check your Python environment.")
        return False
    
    # Set up PostgreSQL
    if not postgres_ok:
        if not setup_postgresql_docker():
            print("❌ PostgreSQL setup failed")
            return False
        setup_steps.append("PostgreSQL")
    
    # Set up Neo4j
    if not neo4j_ok:
        if not setup_neo4j_docker():
            print("❌ Neo4j setup failed")
            return False
        setup_steps.append("Neo4j")
    
    if setup_steps:
        print(f"\n⏳ Waiting for databases to start...")
        await asyncio.sleep(10)  # Give databases time to start
    
    # Re-check connections
    print("\n🔍 Verifying database connections...")
    
    postgres_ok, postgres_msg = await check_postgres_connection()
    print(f"PostgreSQL: {'✅' if postgres_ok else '❌'} {postgres_msg}")
    
    neo4j_ok, neo4j_msg = await check_neo4j_connection()
    print(f"Neo4j: {'✅' if neo4j_ok else '❌'} {neo4j_msg}")
    
    if postgres_ok and neo4j_ok:
        print("\n🎉 All databases are now running!")
        
        # Test agent initialization
        test_success = await test_agent_initialization()
        
        if test_success:
            print("\n✅ Setup completed successfully!")
            print("\n📝 Next steps:")
            print("   1. Run: streamlit run streamlit_app.py")
            print("   2. Go to the Settings page")
            print("   3. Enable agents and select your preferred agent")
            print("   4. Test the agents using the test interface")
        
        return test_success
    else:
        print("\n❌ Some databases are still not accessible")
        print("\n🔧 Manual setup may be required:")
        
        if not postgres_ok:
            print("   PostgreSQL:")
            print("     - Ensure PostgreSQL is running with pgvector extension")
            print("     - Default connection: postgresql://postgres:password@localhost:5432/vectordb")
        
        if not neo4j_ok:
            print("   Neo4j:")
            print("     - Ensure Neo4j is running")
            print("     - Default connection: bolt://localhost:7687 (neo4j/password)")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 