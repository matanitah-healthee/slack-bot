import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Slack bot application."""
    
    # Slack Configuration
    SLACK_BOT_TOKEN: str = os.getenv("SLACK_BOT_TOKEN", "")
    SLACK_SIGNING_SECRET: str = os.getenv("SLACK_SIGNING_SECRET", "")
    SLACK_APP_TOKEN: str = os.getenv("SLACK_APP_TOKEN", "")
    
    # AI API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    OLLAMA_HEALTH_CHECK_INTERVAL: int = int(os.getenv("OLLAMA_HEALTH_CHECK_INTERVAL", "30"))
    
    # Application Settings
    PORT: int = int(os.getenv("PORT", "3000"))
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # AI Model Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        required_slack_vars = [cls.SLACK_BOT_TOKEN, cls.SLACK_SIGNING_SECRET, cls.SLACK_APP_TOKEN]
        has_slack_config = all(bool(var) for var in required_slack_vars)
        
        has_ai_provider = bool(cls.OPENAI_API_KEY or cls.ANTHROPIC_API_KEY or cls.OLLAMA_BASE_URL)
        
        return has_slack_config and has_ai_provider
    
    @classmethod
    def get_ai_provider(cls) -> str:
        """Determine which AI provider to use based on available configuration."""
        if cls.OPENAI_API_KEY:
            return "openai"
        elif cls.ANTHROPIC_API_KEY:
            return "anthropic"
        elif cls.OLLAMA_BASE_URL:
            return "ollama"
        else:
            raise ValueError("No AI provider configured. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_BASE_URL")

# Create a global config instance
config = Config() 