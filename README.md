# 🤖 Slack AI Chatbot with Streamlit Frontend

A powerful Slack chatbot integrated with AI capabilities (OpenAI GPT or Anthropic Claude) and a beautiful Streamlit web dashboard for management and monitoring.

## 🌟 Features

- **AI-Powered Conversations**: Supports both OpenAI and Anthropic APIs with automatic detection
- **Slack Integration**: Responds to direct messages, mentions, and slash commands
- **Web Dashboard**: Streamlit-based frontend for monitoring and management
- **Conversation History**: Tracks and displays conversation history per user
- **Real-time Monitoring**: Live status updates and statistics
- **Configurable Settings**: Easy configuration through environment variables
- **Multiple Run Modes**: Run bot only, frontend only, or both together

## 📋 Prerequisites

- Python 3.8+
- Slack workspace with admin permissions
- One of the following AI providers:
  - OpenAI API key
  - Anthropic API key  
  - Ollama running locally (for local AI models)

## 🚀 Quick Setup

### 1. Create a Slack App

1. Go to [Slack API](https://api.slack.com/apps) and click "Create New App"
2. Choose "From scratch" and give your app a name
3. Select your workspace

### 2. Configure Slack App Permissions

**OAuth & Permissions** → **Bot Token Scopes**:
- `app_mentions:read` - View messages that directly mention your bot
- `channels:read` - View basic information about public channels
- `chat:write` - Send messages as the bot
- `im:read` - View messages in direct messages with the bot
- `im:write` - Start direct messages with people

### 3. Enable Socket Mode

1. Go to **Socket Mode** in your app settings
2. Enable Socket Mode
3. Generate an **App-Level Token** with `connections:write` scope
4. Save the token (starts with `xapp-`)

### 4. Install App to Workspace

1. Go to **Install App** in your app settings
2. Click "Install to Workspace"
3. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### 5. Get Your Signing Secret

1. Go to **Basic Information** in your app settings
2. Copy the **Signing Secret** from the App Credentials section

### 6. Set Up Ollama (Optional - for local AI models)

If you want to use local AI models instead of cloud APIs:

1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai)
2. **Start Ollama**: `ollama serve`
3. **Pull a model**: `ollama pull llama2`
4. **Verify**: `ollama list` to see installed models

## 📦 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example file
   cp env.example .env
   
   # Edit .env with your actual values
   nano .env
   ```

4. **Configure your .env file**:
   ```env
   # Slack Configuration (Required)
   SLACK_BOT_TOKEN=xoxb-your-bot-token-here
   SLACK_SIGNING_SECRET=your-signing-secret-here
   SLACK_APP_TOKEN=xapp-your-app-token-here

       # AI API Keys (Choose one)
    OPENAI_API_KEY=sk-your-openai-api-key-here
    # OR
    ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
    # OR for local models
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama2

   # Optional Configuration
   DEFAULT_MODEL=gpt-3.5-turbo
   MAX_TOKENS=1000
   TEMPERATURE=0.7
   ```

## 🏃‍♂️ Running the Application

### Option 1: Run Everything (Recommended)
```bash
python main.py
```
This starts both the Slack bot and Streamlit dashboard.

### Option 2: Run Slack Bot Only
```bash
python main.py --mode slack
```

### Option 3: Run Streamlit Dashboard Only
```bash
python main.py --mode streamlit
```

### Option 4: Direct Streamlit Run
```bash
streamlit run streamlit_app.py
```

## 🎛️ Using the Dashboard

Once running, access the Streamlit dashboard at `http://localhost:8501`

### Dashboard Features:

1. **Configuration Status**: Check if all environment variables are set correctly
2. **Bot Controls**: Initialize, start, stop, and restart the bot
3. **Real-time Monitoring**: View bot status, AI provider, and statistics
4. **Conversation Viewer**: Browse and manage user conversations
5. **Settings**: Change AI models and test message sending
6. **Logs**: View application status and statistics

## 💬 Using the Slack Bot

### Direct Messages
Send a direct message to the bot - it will respond automatically.

### Channel Mentions
Mention the bot in any channel: `@YourBot hello!`

### Slash Commands
Use the `/chatbot` command: `/chatbot What's the weather like?`

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `SLACK_BOT_TOKEN` | Yes | Bot User OAuth Token | `xoxb-...` |
| `SLACK_SIGNING_SECRET` | Yes | App signing secret | `abc123...` |
| `SLACK_APP_TOKEN` | Yes | App-Level Token | `xapp-...` |
| `OPENAI_API_KEY` | Either* | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Either* | Anthropic API key | `sk-ant-...` |
| `OLLAMA_BASE_URL` | Either* | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | No | Ollama model name | `llama2` |
| `DEFAULT_MODEL` | No | AI model to use | `gpt-3.5-turbo` |
| `MAX_TOKENS` | No | Maximum response tokens | `1000` |
| `TEMPERATURE` | No | AI creativity (0-1) | `0.7` |
| `PORT` | No | Application port | `3000` |
| `STREAMLIT_PORT` | No | Streamlit port | `8501` |
| `DEBUG` | No | Enable debug logging | `True` |

*Either OpenAI API key, Anthropic API key, or Ollama URL is required

### Supported AI Models

**OpenAI:**
- `gpt-3.5-turbo` (default)
- `gpt-4`
- `gpt-4-turbo-preview`

**Anthropic:**
- `claude-3-sonnet-20240229` (default)
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

**Ollama (Local Models):**
- `llama2` (default)
- `llama2:13b`
- `codellama`
- `mistral`
- `neural-chat`
- `starling-lm`
- *Any model available in your local Ollama installation*

## 🐛 Troubleshooting

### Common Issues

1. **"Configuration validation failed"**
   - Check all environment variables are set correctly
   - Ensure API keys are valid and have proper permissions

2. **"Failed to initialize Slack integration"**
   - Verify Slack app is installed to your workspace
   - Check Bot Token Scopes are correctly set
   - Ensure Socket Mode is enabled

3. **"AI service not available"**
   - Verify your AI API key is valid
   - Check API rate limits and billing status

4. **Bot doesn't respond in channels**
   - Invite the bot to the channel: `/invite @YourBot`
   - Ensure the bot has necessary permissions

5. **Socket connection errors**
   - Check internet connectivity
   - Verify App-Level Token is correct and has `connections:write` scope

6. **Ollama connection issues**
   - Ensure Ollama is running: `ollama serve`
   - Verify the Ollama URL is correct (default: `http://localhost:11434`)
   - Check if the specified model is installed: `ollama list`
   - Pull a model if needed: `ollama pull llama2`

### Debug Mode

Run with debug logging:
```bash
python main.py --debug
```

### Testing the Setup

1. Use the dashboard to verify configuration
2. Send a direct message to the bot
3. Check the logs for any errors
4. Use the "Send Test Message" feature in the dashboard

## 🏗️ Project Structure

```
slack-bot/
├── main.py              # Main application entry point
├── config.py            # Configuration management
├── slack_bot.py         # Slack bot implementation
├── ai_service.py        # AI service (OpenAI/Anthropic)
├── slack_integration.py # Integration coordinator
├── streamlit_app.py     # Streamlit dashboard
├── requirements.txt     # Python dependencies
├── env.example          # Environment variables template
└── README.md           # This file
```

## 📝 Development

### Adding New Features

1. **New AI Providers**: Extend `ai_service.py`
2. **New Slack Events**: Add handlers in `slack_bot.py`
3. **Dashboard Pages**: Add to `streamlit_app.py`

### Running Tests

```bash
# Run with debug mode
python main.py --debug

# Test individual components
python -c "from config import config; print(config.validate_config())"
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Run in debug mode to see detailed logs
3. Verify your Slack app configuration
4. Check API key validity and permissions

---

**Happy chatting! 🤖✨** 