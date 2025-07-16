import logging
import threading
import time
from typing import Optional
from config import config
from slack_bot import SlackBot
from ai_service import AIService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackIntegration:
    """Main integration service that coordinates Slack bot and AI service."""
    
    def __init__(self):
        """Initialize the Slack integration."""
        self.ai_service: Optional[AIService] = None
        self.slack_bot: Optional[SlackBot] = None
        self.is_running = False
        self.bot_thread: Optional[threading.Thread] = None
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.ollama_healthy = True
        
    def initialize(self) -> bool:
        """Initialize all services."""
        try:
            # Validate configuration
            if not config.validate_config():
                logger.error("Invalid configuration. Please check your environment variables.")
                return False
            
            # Initialize AI service
            logger.info("Initializing AI service...")
            self.ai_service = AIService()
            
            # Initialize Slack bot with AI service
            logger.info("Initializing Slack bot...")
            self.slack_bot = SlackBot(ai_service=self.ai_service)
            
            logger.info("Slack integration initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Slack integration: {e}")
            return False
    
    def start(self) -> bool:
        """Start the Slack bot service."""
        try:
            if not self.slack_bot:
                logger.error("Slack bot not initialized. Call initialize() first.")
                return False
            
            if self.is_running:
                logger.warning("Slack bot is already running.")
                return True
            
            # Start the bot in a separate thread
            self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
            self.bot_thread.start()
            
            # Start health monitoring for Ollama
            if config.get_ai_provider() == "ollama":
                self.health_monitor_thread = threading.Thread(target=self._monitor_ollama_health, daemon=True)
                self.health_monitor_thread.start()
            
            # Give it a moment to start
            time.sleep(2)
            
            if self.is_running:
                logger.info("Slack bot started successfully!")
                return True
            else:
                logger.error("Failed to start Slack bot.")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Slack bot: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the Slack bot service."""
        try:
            if not self.is_running:
                logger.warning("Slack bot is not running.")
                return True
            
            self.is_running = False
            
            if self.slack_bot:
                self.slack_bot.stop()
            
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=5)
            
            logger.info("Slack bot stopped successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Slack bot: {e}")
            return False
    
    def _run_bot(self):
        """Run the Slack bot in a separate thread."""
        try:
            self.is_running = True
            if self.slack_bot:
                self.slack_bot.start()
            else:
                logger.error("Slack bot is not initialized")
                self.is_running = False
        except Exception as e:
            logger.error(f"Error running Slack bot: {e}")
            self.is_running = False
    
    def _monitor_ollama_health(self):
        """Monitor Ollama health and update bot presence accordingly."""
        logger.info("Starting Ollama health monitoring...")
        
        while self.is_running:
            try:
                if self.ai_service:
                    # Check if Ollama is healthy
                    is_healthy = self.ai_service.health_check()
                    
                    if is_healthy != self.ollama_healthy:
                        self.ollama_healthy = is_healthy
                        self._update_bot_presence()
                        
                        if is_healthy:
                            logger.info("✅ Ollama is healthy - bot is now online")
                        else:
                            logger.warning("❌ Ollama is unhealthy - bot is now away")
                
                # Check at configurable interval
                time.sleep(config.OLLAMA_HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in Ollama health monitoring: {e}")
                time.sleep(config.OLLAMA_HEALTH_CHECK_INTERVAL)
    
    def _update_bot_presence(self):
        """Update bot presence based on Ollama health."""
        try:
            if self.slack_bot and self.slack_bot.app:
                if self.ollama_healthy:
                    # Set bot as active/online
                    self.slack_bot.app.client.users_setPresence(presence="auto")
                    logger.info("Bot presence set to online (Ollama healthy)")
                else:
                    # Set bot as away
                    self.slack_bot.app.client.users_setPresence(presence="away")
                    logger.info("Bot presence set to away (Ollama unhealthy)")
        except Exception as e:
            logger.error(f"Error updating bot presence: {e}")
    
    def get_status(self) -> dict:
        """Get the current status of the integration."""
        status = {
            "is_running": self.is_running,
            "ai_service_initialized": self.ai_service is not None,
            "slack_bot_initialized": self.slack_bot is not None,
            "config_valid": config.validate_config(),
            "ai_provider": config.get_ai_provider() if self.ai_service else None,
            "bot_info": self.slack_bot.get_bot_info() if self.slack_bot else None
        }
        
        # Add Ollama health status for Ollama provider
        if config.get_ai_provider() == "ollama":
            status["ollama_healthy"] = self.ollama_healthy
            status["health_monitoring_active"] = self.health_monitor_thread is not None and self.health_monitor_thread.is_alive()
        
        return status
    
    def get_ai_stats(self) -> dict:
        """Get AI service statistics."""
        if self.ai_service:
            return self.ai_service.get_conversation_stats()
        return {"error": "AI service not initialized"}
    
    def clear_user_conversation(self, user_id: str) -> bool:
        """Clear conversation history for a specific user."""
        if self.ai_service:
            return self.ai_service.clear_conversation(user_id)
        return False
    
    def send_message(self, channel: str, message: str) -> bool:
        """Send a message to a Slack channel."""
        if self.slack_bot:
            result = self.slack_bot.send_message(channel, message)
            return result is not None
        return False
    
    def set_ai_model(self, model: str) -> bool:
        """Set the AI model to use."""
        if self.ai_service:
            return self.ai_service.set_model(model)
        return False
    
    def restart(self) -> bool:
        """Restart the Slack bot service."""
        logger.info("Restarting Slack bot...")
        if self.stop():
            time.sleep(2)
            return self.start()
        return False

# Global integration instance
slack_integration = SlackIntegration() 