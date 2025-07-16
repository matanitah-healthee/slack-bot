#!/usr/bin/env python3
"""
Slack AI Chatbot - Main Application Entry Point

This script provides different ways to run the Slack AI chatbot:
1. Slack bot only (headless mode)
2. Streamlit frontend only
3. Both together (recommended for development)

Usage:
    python main.py --mode slack           # Run only Slack bot
    python main.py --mode streamlit       # Run only Streamlit frontend
    python main.py --mode both            # Run both (default)
    python main.py --help                 # Show help
"""

import argparse
import logging
import multiprocessing
import signal
import sys
import time
import subprocess
from typing import Optional

from config import config
from slack_integration import slack_integration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlackBotApp:
    """Main application class for managing the Slack bot."""
    
    def __init__(self):
        self.slack_process: Optional[multiprocessing.Process] = None
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.running = False
    
    def validate_environment(self) -> bool:
        """Validate that all required environment variables are set."""
        logger.info("Validating environment configuration...")
        
        if not config.validate_config():
            logger.error("Configuration validation failed!")
            logger.error("Please ensure you have:")
            logger.error("1. Set up your Slack app tokens (SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, SLACK_APP_TOKEN)")
            logger.error("2. Set up either OPENAI_API_KEY or ANTHROPIC_API_KEY")
            logger.error("3. Refer to env.example for the complete list of required variables")
            return False
        
        logger.info(f"✅ Configuration valid! Using AI provider: {config.get_ai_provider()}")
        return True
    
    def run_slack_bot(self):
        """Run the Slack bot in a separate process."""
        try:
            logger.info("Starting Slack bot process...")
            
            # Initialize and start the integration
            if not slack_integration.initialize():
                logger.error("Failed to initialize Slack integration")
                return
            
            if not slack_integration.start():
                logger.error("Failed to start Slack bot")
                return
            
            logger.info("Slack bot is running! Press Ctrl+C to stop.")
            
            # Keep the process alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping Slack bot...")
        except Exception as e:
            logger.error(f"Error in Slack bot process: {e}")
        finally:
            slack_integration.stop()
            logger.info("Slack bot stopped.")
    
    def run_streamlit(self):
        """Run the Streamlit frontend."""
        try:
            logger.info("Starting Streamlit frontend...")
            
            # Start Streamlit using subprocess
            cmd = [
                sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.port", str(config.STREAMLIT_PORT),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--server.enableCORS", "false"
            ]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Streamlit frontend started on port {config.STREAMLIT_PORT}")
            logger.info(f"Access the dashboard at: http://localhost:{config.STREAMLIT_PORT}")
            
            # Wait for process to complete
            self.streamlit_process.wait()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping Streamlit...")
        except Exception as e:
            logger.error(f"Error starting Streamlit: {e}")
        finally:
            if self.streamlit_process:
                self.streamlit_process.terminate()
                self.streamlit_process.wait()
            logger.info("Streamlit stopped.")
    
    def run_slack_worker(self):
        """Worker function for running Slack bot in multiprocessing."""
        self.running = True
        self.run_slack_bot()
    
    def run_both(self):
        """Run both Slack bot and Streamlit frontend."""
        logger.info("Starting both Slack bot and Streamlit frontend...")
        
        try:
            # Start Slack bot in a separate process
            self.slack_process = multiprocessing.Process(target=self.run_slack_worker)
            self.slack_process.start()
            
            # Small delay to let Slack bot start
            time.sleep(2)
            
            # Start Streamlit in main process
            self.run_streamlit()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping all services...")
        finally:
            self.stop_all()
    
    def stop_all(self):
        """Stop all running processes."""
        logger.info("Stopping all services...")
        
        self.running = False
        
        # Stop Streamlit
        if self.streamlit_process:
            logger.info("Stopping Streamlit...")
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
        
        # Stop Slack bot
        if self.slack_process and self.slack_process.is_alive():
            logger.info("Stopping Slack bot...")
            self.slack_process.terminate()
            self.slack_process.join(timeout=5)
            if self.slack_process.is_alive():
                self.slack_process.kill()
        
        logger.info("All services stopped.")
    
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_all()
        sys.exit(0)

def print_welcome():
    """Print welcome message and setup instructions."""
    print("=" * 60)
    print("🤖 Slack AI Chatbot")
    print("=" * 60)
    print()
    print("This application creates a Slack chatbot with AI capabilities.")
    print()
    print("Setup Requirements:")
    print("1. Create a Slack app at https://api.slack.com/apps")
    print("2. Set up Bot Token Scopes: app_mentions:read, channels:read, chat:write, im:read, im:write")
    print("3. Enable Socket Mode and generate an App-Level Token")
    print("4. Install the app to your workspace")
    print("5. Set up environment variables (see env.example)")
    print()
    print("Environment Variables Required:")
    print("- SLACK_BOT_TOKEN (starts with xoxb-)")
    print("- SLACK_SIGNING_SECRET")
    print("- SLACK_APP_TOKEN (starts with xapp-)")
    print("- Either OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print()
    print("=" * 60)
    print()

def main():
    """Main entry point."""
    # Print welcome message
    print_welcome()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Slack AI Chatbot Application")
    parser.add_argument(
        "--mode",
        choices=["slack", "streamlit", "both"],
        default="both",
        help="Run mode: slack (bot only), streamlit (frontend only), or both (default)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create application instance
    app = SlackBotApp()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)
    
    # Validate environment
    if not app.validate_environment():
        sys.exit(1)
    
    # Run in specified mode
    try:
        if args.mode == "slack":
            logger.info("Running in Slack bot only mode...")
            app.running = True
            app.run_slack_bot()
        elif args.mode == "streamlit":
            logger.info("Running in Streamlit frontend only mode...")
            app.run_streamlit()
        elif args.mode == "both":
            logger.info("Running in combined mode (Slack bot + Streamlit frontend)...")
            app.run_both()
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        app.stop_all()

if __name__ == "__main__":
    main() 