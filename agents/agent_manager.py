import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .abstract_agent import AbstractAgent
from .rag_bot import RagBot
from .graph_rag_bot import GraphRagBot

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Manages multiple AI agents and routes queries to appropriate agents.
    """
    
    def __init__(self):
        """Initialize the agent manager."""
        self.agents: Dict[str, AbstractAgent] = {}
        self.default_agent: Optional[str] = None
        self.stats = {
            'total_queries': 0,
            'agent_usage': {},
            'last_used': {},
            'errors': {}
        }
        
        # Initialize default agents
        self._setup_default_agents()
    
    def _setup_default_agents(self):
        """Set up default agents."""
        try:
            # Create RAG bot
            rag_bot = RagBot()
            self.register_agent("rag", rag_bot, set_as_default=True)
            
            # Create Graph RAG bot
            graph_rag_bot = GraphRagBot()
            self.register_agent("graph_rag", graph_rag_bot)
            
            logger.info("Default agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up default agents: {e}")
    
    def register_agent(self, agent_id: str, agent: AbstractAgent, set_as_default: bool = False):
        """
        Register an agent with the manager.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: The agent instance
            set_as_default: Whether to set this as the default agent
        """
        self.agents[agent_id] = agent
        self.stats['agent_usage'][agent_id] = 0
        self.stats['errors'][agent_id] = 0
        
        if set_as_default or not self.default_agent:
            self.default_agent = agent_id
        
        logger.info(f"Registered agent: {agent_id} ({agent.name})")
    
    def get_agent(self, agent_id: str) -> Optional[AbstractAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their information."""
        agent_list = []
        for agent_id, agent in self.agents.items():
            info = agent.get_info()
            info['id'] = agent_id
            info['is_default'] = agent_id == self.default_agent
            info['usage_count'] = self.stats['agent_usage'].get(agent_id, 0)
            info['error_count'] = self.stats['errors'].get(agent_id, 0)
            info['last_used'] = self.stats['last_used'].get(agent_id)
            agent_list.append(info)
        return agent_list
    
    async def query(self, message: str, agent_id: Optional[str] = None) -> str:
        """
        Process a query using the specified agent or the default agent.
        
        Args:
            message: The query message
            agent_id: Specific agent to use (optional)
            
        Returns:
            The agent's response
        """
        self.stats['total_queries'] += 1
        
        try:
            # Determine which agent to use
            selected_agent_id = agent_id or self.default_agent
            
            if not selected_agent_id or selected_agent_id not in self.agents:
                return self._generate_error_response("No suitable agent available")
            
            agent = self.agents[selected_agent_id]
            
            # Update usage stats
            self.stats['agent_usage'][selected_agent_id] = self.stats['agent_usage'].get(selected_agent_id, 0) + 1
            self.stats['last_used'][selected_agent_id] = datetime.now().isoformat()
            
            # Process query
            response = await agent.invoke(message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Update error stats
            if selected_agent_id:
                self.stats['errors'][selected_agent_id] = self.stats['errors'].get(selected_agent_id, 0) + 1
            
            return self._generate_error_response(f"Error processing query: {str(e)}")
    
    def _generate_error_response(self, error: str) -> str:
        """Generate a user-friendly error response."""
        available_agents = list(self.agents.keys())
        
        response = f"I encountered an issue: {error}\n\n"
        
        if available_agents:
            response += f"Available agents: {', '.join(available_agents)}\n"
            response += "You can specify an agent by mentioning its name in your query."
        else:
            response += "No agents are currently available. Please contact support."
        
        return response
    
    def set_default_agent(self, agent_id: str) -> bool:
        """Set the default agent."""
        if agent_id in self.agents:
            self.default_agent = agent_id
            logger.info(f"Default agent set to: {agent_id}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_agents': len(self.agents),
            'default_agent': self.default_agent,
            'total_queries': self.stats['total_queries'],
            'agent_usage': self.stats['agent_usage'],
            'last_used': self.stats['last_used'],
            'errors': self.stats['errors']
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of all agents."""
        health_status = {
            'overall_healthy': True,
            'agents': {}
        }
        
        for agent_id, agent in self.agents.items():
            # For now, assume agents are healthy if they're registered
            # In the future, agents could implement their own health_check method
            agent_health = hasattr(agent, 'health_check') and callable(getattr(agent, 'health_check'))
            health_status['agents'][agent_id] = {
                'healthy': True,  # Default to healthy for basic agents
                'name': agent.name,
                'type': agent.__class__.__name__
            }
        
        return health_status

# Global agent manager instance
agent_manager = AgentManager() 