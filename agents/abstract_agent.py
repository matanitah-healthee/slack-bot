import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AbstractAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent."""
        pass
    
    @abstractmethod
    async def invoke(self, message: str) -> str:
        """Invoke the agent."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }