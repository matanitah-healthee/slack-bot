import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    """AI service that supports both direct AI providers and agents."""
    
    def __init__(self):
        """Initialize the AI service with the appropriate provider."""
        self.provider = config.get_ai_provider()
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.client = None
        
        # Agent system
        self.agent_manager = None
        self.use_agents = False
        self.selected_agent = None
        
        # Initialize the appropriate client
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")
        
        # Initialize agent manager
        self._init_agents()
        
        logger.info(f"AI Service initialized with provider: {self.provider}")
        if self.agent_manager:
            logger.info("Agent system available")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.DEFAULT_MODEL or "gpt-3.5-turbo"
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model = config.DEFAULT_MODEL or "claude-3-sonnet-20240229"
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama.Client(host=config.OLLAMA_BASE_URL)
            self.model = config.OLLAMA_MODEL
        except ImportError:
            raise ImportError("Ollama package not installed. Run: pip install ollama")
    
    def _init_agents(self):
        """Initialize the agent system."""
        try:
            from agents.agent_manager import agent_manager
            self.agent_manager = agent_manager
            logger.info("Agent system initialized successfully")
        except Exception as e:
            logger.warning(f"Agent system not available: {e}")
            self.agent_manager = None
    
    def get_response(self, message: str, user_id: str, context: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """Get AI response for a message from a specific user."""
        try:
            # Check if we should use agents
            if (self.use_agents or agent_id) and self.agent_manager:
                return self._get_agent_response(message, user_id, agent_id)
            
            # Use direct AI provider
            return self._get_direct_ai_response(message, user_id, context)
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def _get_agent_response(self, message: str, user_id: str, agent_id: Optional[str] = None) -> str:
        """Get response using the agent system."""
        try:
            # Use specific agent or selected agent or default
            target_agent = agent_id or self.selected_agent
            
            # Since agent manager query is async, we need to handle it properly
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, create a task
                task = asyncio.create_task(self.agent_manager.query(message, target_agent))
                # For synchronous interface, we need to work around this
                # For now, return a simple response indicating agent processing
                return f"[Agent Processing] Your message is being processed by the agent system. In a future update, this will provide the full agent response."
            except RuntimeError:
                # No event loop, we can use asyncio.run
                return asyncio.run(self.agent_manager.query(message, target_agent))
                
        except Exception as e:
            logger.error(f"Error getting agent response: {e}")
            return f"Agent system error: {str(e)}"
    
    def _get_direct_ai_response(self, message: str, user_id: str, context: Optional[str] = None) -> str:
        """Get response using direct AI provider."""
        # Get or create conversation history for user
        conversation = self.get_conversation(user_id)
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        conversation.append(user_message)
        
        # Prepare messages for AI
        messages = self._prepare_messages(conversation, context)
        
        # Get AI response based on provider
        if self.provider == "openai":
            response_text = self._get_openai_response(messages)
        elif self.provider == "anthropic":
            response_text = self._get_anthropic_response(messages)
        elif self.provider == "ollama":
            response_text = self._get_ollama_response(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Add AI response to conversation
        ai_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        }
        conversation.append(ai_message)
        
        # Trim conversation if it gets too long
        self._trim_conversation(user_id)
        
        return response_text
    
    def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _get_anthropic_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from Anthropic."""
        try:
            # Convert messages format for Anthropic
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _get_ollama_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from Ollama."""
        try:
            # Convert messages format for Ollama
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            
            prompt += "Assistant: "
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": config.TEMPERATURE,
                    "num_predict": config.MAX_TOKENS,
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _prepare_messages(self, conversation: List[Dict[str, Any]], context: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for AI API."""
        messages = []
        
        # Add system message
        system_content = "You are a helpful AI assistant integrated into Slack. "
        system_content += "Provide concise, helpful responses to user questions. "
        system_content += "Be friendly and professional. Keep responses reasonably short for chat."
        
        if context:
            system_content += f"\n\nAdditional context: {context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (exclude timestamp for API)
        for msg in conversation[-10:]:  # Last 10 messages to manage context length
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return messages
    
    def get_conversation(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        return self.conversations[user_id]
    
    def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation history for a user."""
        if user_id in self.conversations:
            self.conversations[user_id] = []
            return True
        return False
    
    def _trim_conversation(self, user_id: str, max_messages: int = 20):
        """Trim conversation history to prevent memory issues."""
        if user_id in self.conversations:
            conversation = self.conversations[user_id]
            if len(conversation) > max_messages:
                # Keep the most recent messages
                self.conversations[user_id] = conversation[-max_messages:]
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversations."""
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv) for conv in self.conversations.values())
        
        return {
            "provider": self.provider,
            "model": self.model,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "active_users": list(self.conversations.keys())
        }
    
    def set_model(self, model: str) -> bool:
        """Set the AI model to use."""
        try:
            self.model = model
            logger.info(f"Model changed to: {model}")
            return True
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if the AI service is healthy and can respond."""
        try:
            if self.provider == "openai":
                # Test OpenAI connection
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                return True
            elif self.provider == "anthropic":
                # Test Anthropic connection
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
                return True
            elif self.provider == "ollama":
                # Test Ollama connection
                response = self.client.generate(
                    model=self.model,
                    prompt="test",
                    options={"num_predict": 1}
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"AI service health check failed: {e}")
            return False
    
    # Agent management methods
    def set_use_agents(self, use_agents: bool) -> bool:
        """Enable or disable agent usage."""
        if self.agent_manager:
            self.use_agents = use_agents
            logger.info(f"Agent usage set to: {use_agents}")
            return True
        return False
    
    def set_selected_agent(self, agent_id: Optional[str]) -> bool:
        """Set the selected agent."""
        if self.agent_manager:
            self.selected_agent = agent_id
            logger.info(f"Selected agent set to: {agent_id}")
            return True
        return False
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents."""
        if self.agent_manager:
            try:
                return self.agent_manager.list_agents()
            except Exception as e:
                logger.error(f"Error getting agents: {e}")
                return []
        return []
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent usage statistics."""
        if self.agent_manager:
            try:
                return self.agent_manager.get_stats()
            except Exception as e:
                logger.error(f"Error getting agent stats: {e}")
                return {}
        return {} 