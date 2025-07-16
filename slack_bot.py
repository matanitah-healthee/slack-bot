import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackBot:
    """Slack bot class that handles all Slack interactions."""
    
    def __init__(self, ai_service=None):
        """Initialize the Slack bot with AI service."""
        self.ai_service = ai_service
        
        # Initialize Slack app
        self.app = App(
            token=config.SLACK_BOT_TOKEN,
            signing_secret=config.SLACK_SIGNING_SECRET
        )
        
        # Set up event handlers
        self._setup_handlers()
        
        # Socket Mode Handler for development
        self.handler = SocketModeHandler(self.app, config.SLACK_APP_TOKEN)
    
    def _setup_handlers(self):
        """Set up all Slack event handlers."""
        
        # Handle direct messages and mentions
        @self.app.event("message")
        def handle_message_events(event, say, logger):
            """Handle incoming messages."""
            try:
                # Skip bot messages and messages without text
                if event.get("subtype") == "bot_message" or not event.get("text"):
                    return
                
                user_id = event.get("user")
                text = event.get("text", "")
                channel = event.get("channel")
                
                # Check if bot is mentioned or it's a DM
                bot_user_id = self.app.client.auth_test()["user_id"]
                is_dm = channel.startswith("D")
                is_mentioned = f"<@{bot_user_id}>" in text
                
                if is_dm or is_mentioned:
                    # Remove bot mention from text
                    if is_mentioned:
                        text = text.replace(f"<@{bot_user_id}>", "").strip()
                    
                    # Get AI response
                    if self.ai_service:
                        try:
                            response = self.ai_service.get_response(text, user_id)
                            say(response)
                        except Exception as e:
                            logger.error(f"Error getting AI response: {e}")
                            say("Sorry, I encountered an error processing your request. Please try again.")
                    else:
                        say("AI service is not available. Please check the configuration.")
                        
            except Exception as e:
                logger.error(f"Error handling message: {e}")
        
        # Handle app mentions
        @self.app.event("app_mention")
        def handle_app_mentions(event, say, logger):
            """Handle app mentions."""
            try:
                text = event.get("text", "")
                user_id = event.get("user")
                
                # Remove bot mention from text
                bot_user_id = self.app.client.auth_test()["user_id"]
                text = text.replace(f"<@{bot_user_id}>", "").strip()
                
                if self.ai_service:
                    try:
                        response = self.ai_service.get_response(text, user_id)
                        say(response)
                    except Exception as e:
                        logger.error(f"Error getting AI response: {e}")
                        say("Sorry, I encountered an error processing your request. Please try again.")
                else:
                    say("AI service is not available. Please check the configuration.")
                    
            except Exception as e:
                logger.error(f"Error handling app mention: {e}")
        
        # Handle slash commands
        @self.app.command("/chatbot")
        def handle_chatbot_command(ack, respond, command):
            """Handle /chatbot slash command."""
            ack()
            try:
                text = command.get("text", "").strip()
                user_id = command.get("user_id")
                
                if not text:
                    respond("Please provide a message. Usage: `/chatbot your message here`")
                    return
                
                if self.ai_service:
                    try:
                        response = self.ai_service.get_response(text, user_id)
                        respond(response)
                    except Exception as e:
                        logger.error(f"Error getting AI response: {e}")
                        respond("Sorry, I encountered an error processing your request. Please try again.")
                else:
                    respond("AI service is not available. Please check the configuration.")
                    
            except Exception as e:
                logger.error(f"Error handling slash command: {e}")
                respond("Sorry, an error occurred processing your command.")
    
    def start(self):
        """Start the Slack bot."""
        logger.info("Starting Slack bot...")
        self.handler.start()
    
    def stop(self):
        """Stop the Slack bot."""
        logger.info("Stopping Slack bot...")
        self.handler.close()
    
    def send_message(self, channel, text):
        """Send a message to a specific channel."""
        try:
            result = self.app.client.chat_postMessage(
                channel=channel,
                text=text
            )
            return result
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
    
    def get_bot_info(self):
        """Get bot information."""
        try:
            auth_result = self.app.client.auth_test()
            return {
                "user_id": auth_result["user_id"],
                "bot_id": auth_result["bot_id"],
                "team": auth_result["team"],
                "user": auth_result["user"]
            }
        except Exception as e:
            logger.error(f"Error getting bot info: {e}")
            return None 