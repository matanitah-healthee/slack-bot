import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    """AI service that supports both OpenAI and Anthropic APIs."""
    
    def __init__(self):
        """Initialize the AI service with the appropriate provider."""
        self.provider = config.get_ai_provider()
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.client = None
        
        # Initialize the appropriate client
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")
        
        logger.info(f"AI Service initialized with provider: {self.provider}")
    
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
    
    def get_response(self, message: str, user_id: str, context: Optional[str] = None) -> str:
        """Get AI response for a message from a specific user."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
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