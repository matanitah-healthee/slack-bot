#!/usr/bin/env python3
"""
Test script for the agent system functionality.

This script demonstrates:
1. Agent initialization
2. Agent listing and information
3. Agent response testing
4. Agent selection and switching
"""

import asyncio
import sys
import logging
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_system():
    """Test the agent system functionality."""
    print("🤖 Testing Agent System")
    print("=" * 50)
    
    try:
        # Import agent manager
        from agents.agent_manager import agent_manager
        
        print("✅ Agent manager imported successfully")
        
        # List available agents
        agents = agent_manager.list_agents()
        print(f"\n📋 Available Agents ({len(agents)}):")
        
        for agent in agents:
            status = "🌟 Default" if agent.get('is_default') else "   Regular"
            print(f"  {status} | {agent['id']} - {agent['name']}")
            print(f"          | Type: {agent['type']}")
            print(f"          | Description: {agent['description']}")
            if agent.get('capabilities'):
                print(f"          | Capabilities: {', '.join(agent['capabilities'])}")
            print()
        
        # Test database-specific queries for Healthee content
        test_queries = [
            "What is Healthee?",
            "Who does Healthee serve?",
            "What are Healthee's key features?",
            "Tell me about Zoe"
        ]
        
        print("🧪 Testing Agent Responses with Database Integration")
        print("-" * 30)
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            
            # Test with default agent
            try:
                response = await agent_manager.query(query)
                print(f"🤖 Default Agent Response:")
                print(f"   {response[:150]}{'...' if len(response) > 150 else ''}")
            except Exception as e:
                print(f"❌ Error with default agent: {e}")
            
            # Test with specific agents
            for agent in agents:
                agent_id = agent['id']
                try:
                    response = await agent_manager.query(query, agent_id=agent_id)
                    print(f"🤖 {agent['name']} Response:")
                    print(f"   {response[:150]}{'...' if len(response) > 150 else ''}")
                except Exception as e:
                    print(f"❌ Error with {agent['name']}: {e}")
            
            print()
        
        # Test individual agent statistics
        print("📊 Individual Agent Statistics:")
        for agent in agents:
            agent_obj = agent_manager.get_agent(agent['id'])
            if agent_obj and hasattr(agent_obj, 'get_stats') and callable(getattr(agent_obj, 'get_stats')):
                try:
                    stats = await agent_obj.get_stats()  # type: ignore
                    print(f"   {agent['name']}: {stats}")
                except Exception as e:
                    print(f"   {agent['name']}: Error getting stats - {e}")
            else:
                print(f"   {agent['name']}: No detailed stats available")
        
        # Get agent statistics
        stats = agent_manager.get_stats()
        print("\n📊 Overall Agent Statistics:")
        print(f"   Total Agents: {stats.get('total_agents', 0)}")
        print(f"   Default Agent: {stats.get('default_agent', 'None')}")
        print(f"   Total Queries: {stats.get('total_queries', 0)}")
        
        if stats.get('agent_usage'):
            print("   Usage by Agent:")
            for agent_id, count in stats['agent_usage'].items():
                print(f"     - {agent_id}: {count} queries")
        
        # Health check
        health = agent_manager.health_check()
        print(f"\n🏥 System Health: {'✅ Healthy' if health.get('overall_healthy') else '❌ Unhealthy'}")
        
        print("\n✅ Agent system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing agent system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_service_integration():
    """Test AI service integration with agents."""
    print("\n🔗 Testing AI Service Integration")
    print("=" * 50)
    
    try:
        from ai_service import AIService
        
        # Create AI service
        ai_service = AIService()
        print("✅ AI Service created successfully")
        
        # Check agent availability
        available_agents = ai_service.get_available_agents()
        print(f"📋 Agents available in AI Service: {len(available_agents)}")
        
        for agent in available_agents:
            print(f"   - {agent['id']}: {agent['name']}")
        
        # Test agent selection
        if available_agents:
            first_agent = available_agents[0]
            success = ai_service.set_selected_agent(first_agent['id'])
            print(f"🎯 Set selected agent to '{first_agent['id']}': {'✅' if success else '❌'}")
            
            # Enable agents
            success = ai_service.set_use_agents(True)
            print(f"🔄 Enabled agent usage: {'✅' if success else '❌'}")
            
            # Test response
            test_message = "Hello, can you help me?"
            print(f"\n📝 Testing message: '{test_message}'")
            response = ai_service.get_response(test_message, "test_user")
            print(f"🤖 Response: {response[:150]}{'...' if len(response) > 150 else ''}")
        
        # Get agent stats
        agent_stats = ai_service.get_agent_stats()
        print(f"\n📊 Agent Stats: {agent_stats}")
        
        print("\n✅ AI Service integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI service integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_connectivity():
    """Test database connectivity for both pgvector and Neo4j."""
    print("\n🔗 Testing Database Connectivity")
    print("=" * 50)
    
    test_results = {
        "pgvector": False,
        "neo4j": False
    }
    
    # Test pgvector (PostgreSQL)
    try:
        from utils.vector_store import VectorStore
        
        vector_store = VectorStore("test_connection")
        success = await vector_store.initialize()
        
        if success:
            print("✅ pgvector (PostgreSQL) connection successful")
            doc_count = await vector_store.get_document_count()
            print(f"   Documents in database: {doc_count}")
            test_results["pgvector"] = True
            await vector_store.close()
        else:
            print("❌ pgvector (PostgreSQL) connection failed")
            
    except Exception as e:
        print(f"❌ pgvector test error: {e}")
    
    # Test Neo4j
    try:
        from utils.graph_store import GraphStore
        
        graph_store = GraphStore()
        success = await graph_store.initialize()
        
        if success:
            print("✅ Neo4j connection successful")
            node_counts = await graph_store.get_node_count()
            rel_count = await graph_store.get_relationship_count()
            print(f"   Nodes: {node_counts}, Relationships: {rel_count}")
            test_results["neo4j"] = True
            await graph_store.close()
        else:
            print("❌ Neo4j connection failed")
            
    except Exception as e:
        print(f"❌ Neo4j test error: {e}")
    
    return test_results

async def main():
    """Run all tests."""
    print("🚀 Starting Agent System Tests with Database Integration")
    print("=" * 60)
    
    # Test database connectivity first
    db_test_results = await test_database_connectivity()
    
    # Test agent system
    agent_test_result = await test_agent_system()
    
    # Test AI service integration
    ai_service_test_result = test_ai_service_integration()
    
    print("\n📝 Test Summary")
    print("=" * 30)
    print(f"PostgreSQL/pgvector: {'✅ PASS' if db_test_results['pgvector'] else '❌ FAIL'}")
    print(f"Neo4j: {'✅ PASS' if db_test_results['neo4j'] else '❌ FAIL'}")
    print(f"Agent System: {'✅ PASS' if agent_test_result else '❌ FAIL'}")
    print(f"AI Service Integration: {'✅ PASS' if ai_service_test_result else '❌ FAIL'}")
    
    overall_success = (
        db_test_results['pgvector'] and 
        db_test_results['neo4j'] and
        agent_test_result and 
        ai_service_test_result
    )
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if not overall_success:
        print("\n⚠️  Setup Requirements:")
        if not db_test_results['pgvector']:
            print("   • Start PostgreSQL with pgvector extension")
            print("   • Run: docker-compose up -d (if using provided docker-compose.yml)")
        if not db_test_results['neo4j']:
            print("   • Start Neo4j database")
            print("   • Default: bolt://localhost:7687 with neo4j/password credentials")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 