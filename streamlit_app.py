import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any
from config import config
from slack_integration import slack_integration

# Page configuration
st.set_page_config(
    page_title="Slack AI Chatbot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'integration_initialized' not in st.session_state:
        st.session_state.integration_initialized = False
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None

def check_configuration():
    """Check if the configuration is valid."""
    config_status = {
        "Slack Bot Token": bool(config.SLACK_BOT_TOKEN),
        "Slack Signing Secret": bool(config.SLACK_SIGNING_SECRET),
        "Slack App Token": bool(config.SLACK_APP_TOKEN),
        "OpenAI API Key": bool(config.OPENAI_API_KEY),
        "Anthropic API Key": bool(config.ANTHROPIC_API_KEY),
        "Ollama URL": bool(config.OLLAMA_BASE_URL),
    }
    
    has_ai_provider = config_status["OpenAI API Key"] or config_status["Anthropic API Key"] or config_status["Ollama URL"]
    has_slack_config = all([
        config_status["Slack Bot Token"],
        config_status["Slack Signing Secret"],
        config_status["Slack App Token"]
    ])
    
    return config_status, has_slack_config and has_ai_provider

def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("🤖 Slack AI Bot")
    
    # Configuration status
    config_status, is_valid = check_configuration()
    
    with st.sidebar.expander("📊 Configuration Status", expanded=True):
        for key, value in config_status.items():
            icon = "✅" if value else "❌"
            st.write(f"{icon} {key}")
        
        if is_valid:
            st.success("Configuration is valid!")
        else:
            st.error("Configuration incomplete. Check env.example file.")
    
    # Bot controls
    st.sidebar.header("🎮 Bot Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Initialize", use_container_width=True):
            if is_valid:
                with st.spinner("Initializing..."):
                    success = slack_integration.initialize()
                    if success:
                        st.session_state.integration_initialized = True
                        st.success("Initialized!")
                    else:
                        st.error("Initialization failed!")
            else:
                st.error("Fix configuration first!")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    # Bot status and controls
    if st.session_state.integration_initialized:
        status = slack_integration.get_status()
        
        if status["is_running"]:
            if st.sidebar.button("⏹️ Stop Bot", use_container_width=True):
                slack_integration.stop()
                st.rerun()
        else:
            if st.sidebar.button("▶️ Start Bot", use_container_width=True):
                slack_integration.start()
                st.rerun()
        
        if st.sidebar.button("🔄 Restart Bot", use_container_width=True):
            slack_integration.restart()
            st.rerun()
    
    # Navigation
    st.sidebar.header("📱 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Conversations", "Settings", "Logs"]
    )
    
    return page

def render_dashboard():
    """Render the main dashboard page."""
    st.title("🤖 Slack AI Chatbot Dashboard")
    
    if not st.session_state.integration_initialized:
        st.warning("⚠️ Please initialize the bot first using the sidebar controls.")
        return
    
    # Get status
    status = slack_integration.get_status()
    ai_stats = slack_integration.get_ai_stats()
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_icon = "🟢" if status["is_running"] else "🔴"
        st.metric("Bot Status", f"{status_icon} {'Running' if status['is_running'] else 'Stopped'}")
    
    with col2:
        provider = status.get("ai_provider", "Unknown")
        st.metric("AI Provider", provider.title() if provider else "Unknown")
    
    with col3:
        total_conversations = ai_stats.get("total_conversations", 0)
        st.metric("Total Conversations", total_conversations)
    
    with col4:
        total_messages = ai_stats.get("total_messages", 0)
        st.metric("Total Messages", total_messages)
    
    # Ollama health status for Ollama provider
    if status.get("ai_provider") == "ollama":
        st.subheader("🏥 Ollama Health Status")
        if slack_integration.ai_service:
            try:
                is_healthy = slack_integration.ai_service.health_check()
                if is_healthy:
                    st.success("✅ Ollama is healthy and responding")
                else:
                    st.error("❌ Ollama is not responding - bot will appear offline")
            except Exception as e:
                st.warning(f"⚠️ Could not check Ollama health: {e}")
        else:
            st.info("AI service not initialized")
    
    # Bot information
    if status["bot_info"]:
        st.subheader("🤖 Bot Information")
        bot_info = status["bot_info"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Bot User ID:** {bot_info.get('user_id', 'N/A')}")
            st.info(f"**Bot ID:** {bot_info.get('bot_id', 'N/A')}")
        with col2:
            st.info(f"**Team:** {bot_info.get('team', 'N/A')}")
            st.info(f"**User:** {bot_info.get('user', 'N/A')}")
    
    # AI Model information
    st.subheader("🧠 AI Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Current Model:** {ai_stats.get('model', 'Unknown')}")
        st.info(f"**Max Tokens:** {config.MAX_TOKENS}")
    
    with col2:
        st.info(f"**Temperature:** {config.TEMPERATURE}")
        st.info(f"**Provider:** {ai_stats.get('provider', 'Unknown').title()}")
    
    # Agent Statistics (if available)
    if slack_integration.ai_service:
        agent_stats = slack_integration.ai_service.get_agent_stats()
        available_agents = slack_integration.ai_service.get_available_agents()
        
        if agent_stats or available_agents:
            st.subheader("🤖 Agent System")
            
            # Agent overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Agents", agent_stats.get("total_agents", 0))
            
            with col2:
                default_agent = agent_stats.get("default_agent", "None")
                st.metric("Default Agent", default_agent if default_agent else "None")
            
            with col3:
                st.metric("Agent Queries", agent_stats.get("total_queries", 0))
            
            with col4:
                agent_mode = "🟢 Enabled" if slack_integration.ai_service.use_agents else "🔴 Disabled"
                st.metric("Agent Status", agent_mode)
            
            # Agent usage chart
            if agent_stats.get("agent_usage"):
                usage_data = agent_stats["agent_usage"]
                if any(count > 0 for count in usage_data.values()):
                    st.write("**Agent Usage Distribution:**")
                    usage_df = pd.DataFrame([
                        {"Agent": agent_id, "Usage Count": count}
                        for agent_id, count in usage_data.items()
                        if count > 0
                    ])
                    st.bar_chart(usage_df.set_index("Agent"))
            
            # Current agent selection
            if slack_integration.ai_service.use_agents:
                current_agent = slack_integration.ai_service.selected_agent or "Auto-select"
                st.info(f"**Currently Selected Agent:** {current_agent}")
            
            # Agent details in expander
            if available_agents:
                with st.expander("🔍 View Agent Details"):
                    for agent in available_agents:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            agent_name = f"**{agent['name']}** ({agent['id']})"
                            if agent.get('is_default'):
                                agent_name += " 🌟"
                            st.write(agent_name)
                            st.caption(agent['description'])
                        
                        with col2:
                            st.metric("Usage", agent.get('usage_count', 0))
                        
                        with col3:
                            st.metric("Errors", agent.get('error_count', 0))
                        
                        st.divider()

def render_conversations():
    """Render the conversations page."""
    st.title("💬 Conversations")
    
    if not st.session_state.integration_initialized:
        st.warning("⚠️ Please initialize the bot first using the sidebar controls.")
        return
    
    ai_stats = slack_integration.get_ai_stats()
    active_users = ai_stats.get("active_users", [])
    
    if not active_users:
        st.info("No conversations found. Start chatting with the bot in Slack!")
        return
    
    # User selection
    selected_user = st.selectbox("Select User", active_users)
    
    if selected_user:
        st.session_state.selected_user = selected_user
        
        # Display conversation
        if slack_integration.ai_service:
            conversation = slack_integration.ai_service.get_conversation(selected_user)
            
            if conversation:
                st.subheader(f"Conversation with {selected_user}")
                
                # Conversation display
                for i, message in enumerate(conversation):
                    timestamp = message.get("timestamp", "")
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    if role == "user":
                        st.chat_message("user").write(f"**{timestamp}**\n\n{content}")
                    elif role == "assistant":
                        st.chat_message("assistant").write(f"**{timestamp}**\n\n{content}")
                
                # Clear conversation button
                if st.button(f"🗑️ Clear conversation with {selected_user}"):
                    success = slack_integration.clear_user_conversation(selected_user)
                    if success:
                        st.success("Conversation cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear conversation.")
            else:
                st.info(f"No messages found for {selected_user}")

def render_settings():
    """Render the settings page."""
    st.title("⚙️ Settings")
    
    if not st.session_state.integration_initialized:
        st.warning("⚠️ Please initialize the bot first using the sidebar controls.")
        return
    
    # Agent Selection
    st.subheader("🤖 Agent Selection")
    
    if slack_integration.ai_service:
        # Check if agents are available
        available_agents = slack_integration.ai_service.get_available_agents()
        
        if available_agents:
            # Agent usage toggle
            use_agents = st.checkbox("Use AI Agents", 
                                   value=slack_integration.ai_service.use_agents,
                                   help="Enable AI agents for specialized responses")
            
            if st.button("Toggle Agent Usage"):
                success = slack_integration.ai_service.set_use_agents(use_agents)
                if success:
                    st.success(f"Agent usage {'enabled' if use_agents else 'disabled'}")
                    st.rerun()
                else:
                    st.error("Failed to update agent usage")
            
            # Agent selection
            if use_agents or slack_integration.ai_service.use_agents:
                st.write("**Available Agents:**")
                
                agent_options = ["Default (Auto-select)"] + [f"{agent['id']} - {agent['name']}" for agent in available_agents]
                current_selection = slack_integration.ai_service.selected_agent
                
                # Find current index
                current_index = 0
                if current_selection:
                    for i, agent in enumerate(available_agents):
                        if agent['id'] == current_selection:
                            current_index = i + 1
                            break
                
                selected_option = st.selectbox("Select Agent", agent_options, index=current_index)
                
                if st.button("Update Selected Agent"):
                    if selected_option and selected_option == "Default (Auto-select)":
                        agent_id = None
                    elif selected_option and " - " in selected_option:
                        # Extract agent ID from the option string
                        agent_id = selected_option.split(" - ")[0]
                    else:
                        agent_id = None
                    
                    success = slack_integration.ai_service.set_selected_agent(agent_id)
                    if success:
                        st.success(f"Selected agent updated to: {selected_option}")
                        st.rerun()
                    else:
                        st.error("Failed to update selected agent")
                
                # Display agent information
                st.write("**Agent Details:**")
                for agent in available_agents:
                    with st.expander(f"{agent['name']} ({agent['id']})"):
                        st.write(f"**Description:** {agent['description']}")
                        st.write(f"**Type:** {agent['type']}")
                        st.write(f"**Usage Count:** {agent.get('usage_count', 0)}")
                        st.write(f"**Error Count:** {agent.get('error_count', 0)}")
                        if agent.get('last_used'):
                            st.write(f"**Last Used:** {agent['last_used']}")
                        if agent.get('capabilities'):
                            st.write(f"**Capabilities:** {', '.join(agent['capabilities'])}")
                        
                        # Default agent indicator
                        if agent.get('is_default'):
                            st.success("🌟 Default Agent")
        else:
            st.info("No AI agents are currently available. Using direct AI provider.")
    
    st.divider()
    
    # AI Model Settings
    st.subheader("🧠 AI Model Settings")
    
    provider = config.get_ai_provider()
    
    # Model selection based on provider
    if provider == "openai":
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    elif provider == "anthropic":
        models = ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    elif provider == "ollama":
        models = ["llama2", "llama2:13b", "codellama", "mistral", "neural-chat", "starling-lm"]
    else:
        models = [config.DEFAULT_MODEL]
    
    current_model = config.DEFAULT_MODEL
    selected_model = st.selectbox("AI Model", models, index=models.index(current_model) if current_model in models else 0)
    
    if st.button("Update Model"):
        if selected_model:
            success = slack_integration.set_ai_model(selected_model)
            if success:
                st.success(f"Model updated to {selected_model}")
            else:
                st.error("Failed to update model")
        else:
            st.error("Please select a model")
    
    # Configuration display
    st.subheader("📋 Current Configuration")
    
    config_data = {
        "Setting": ["AI Provider", "Current Model", "Max Tokens", "Temperature", "Debug Mode"],
        "Value": [
            provider.title(),
            selected_model,
            config.MAX_TOKENS,
            config.TEMPERATURE,
            config.DEBUG
        ]
    }
    
    df = pd.DataFrame(config_data)
    st.table(df)
    
    # Agent Test Interface
    st.subheader("🧪 Test Agent Responses")
    
    if slack_integration.ai_service and available_agents:
        test_message = st.text_input("Test Message", placeholder="Enter a message to test agent responses...")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Agent selection for testing
            test_agent_options = ["Default AI Provider"] + [f"{agent['id']} - {agent['name']}" for agent in available_agents]
            selected_test_agent = st.selectbox("Select Agent/Provider for Testing", test_agent_options)
        
        with col2:
            st.write("")  # Spacer
            if st.button("🧪 Test Response", use_container_width=True):
                if test_message:
                    with st.spinner("Getting response..."):
                        try:
                            if selected_test_agent == "Default AI Provider":
                                # Use direct AI provider
                                response = slack_integration.ai_service._get_direct_ai_response(test_message, "test_user")
                            elif selected_test_agent and " - " in selected_test_agent:
                                # Extract agent ID and use agent
                                agent_id = selected_test_agent.split(" - ")[0]
                                response = slack_integration.ai_service.get_response(test_message, "test_user", agent_id=agent_id)
                            else:
                                response = "Error: Invalid agent selection"
                            
                            st.success("Response received!")
                            st.write("**Response:**")
                            st.write(response)
                            
                        except Exception as e:
                            st.error(f"Error getting response: {str(e)}")
                else:
                    st.warning("Please enter a test message.")
    else:
        st.info("Agent testing is not available. No agents found.")
    
    st.divider()
    
    # Send test message
    st.subheader("📤 Send Test Message")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        channel = st.text_input("Channel ID", placeholder="C1234567890")
        message = st.text_area("Message", placeholder="Enter your test message here...")
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("Send Message", use_container_width=True):
            if channel and message:
                success = slack_integration.send_message(channel, message)
                if success:
                    st.success("Message sent!")
                else:
                    st.error("Failed to send message")
            else:
                st.error("Please provide both channel and message")

def render_logs():
    """Render the logs page."""
    st.title("📋 Logs")
    
    st.info("Logs are displayed in the terminal where you're running the application.")
    
    # Display basic status
    if st.session_state.integration_initialized:
        status = slack_integration.get_status()
        
        st.subheader("Current Status")
        st.json(status)
        
        st.subheader("AI Statistics")
        ai_stats = slack_integration.get_ai_stats()
        st.json(ai_stats)
    else:
        st.warning("Initialize the bot to see status information.")

def main():
    """Main application function."""
    initialize_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Conversations":
        render_conversations()
    elif page == "Settings":
        render_settings()
    elif page == "Logs":
        render_logs()

if __name__ == "__main__":
    main() 