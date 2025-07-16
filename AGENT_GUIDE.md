# 🤖 Agent Selection Guide

This guide explains how to use the new agent selection feature in the Slack AI Bot Dashboard.

## Overview

The agent system allows you to choose between different AI agents, each specialized for different types of tasks. Instead of using only direct AI providers (OpenAI, Anthropic, Ollama), you can now route your queries through specialized agents.

## Available Agents

### 1. RAG Bot (`rag`)
- **Description**: Retrieval-Augmented Generation Bot
- **Purpose**: Searches through documents and knowledge bases to provide contextual responses
- **Best for**: Questions requiring specific information retrieval, documentation queries
- **Capabilities**: `document_search`, `contextual_responses`, `knowledge_retrieval`

### 2. Graph RAG Bot (`graph_rag`)
- **Description**: Graph-based Retrieval-Augmented Generation Bot  
- **Purpose**: Analyzes relationships between concepts and provides comprehensive responses
- **Best for**: Complex questions requiring understanding of connections and relationships
- **Capabilities**: `graph_traversal`, `relationship_analysis`, `connected_knowledge_retrieval`

## Using Agent Selection in Streamlit Dashboard

### 1. Accessing Agent Settings

1. Open the Streamlit dashboard at `http://localhost:8501`
2. Navigate to the **Settings** page
3. Look for the **🤖 Agent Selection** section

### 2. Enabling Agents

1. Check the **"Use AI Agents"** checkbox
2. Click **"Toggle Agent Usage"** to enable the agent system
3. The dashboard will show available agents

### 3. Selecting an Agent

1. From the **"Select Agent"** dropdown, choose:
   - **Default (Auto-select)**: Let the system choose the best agent
   - **rag - RAG Bot**: Use the RAG bot specifically
   - **graph_rag - Graph RAG Bot**: Use the Graph RAG bot specifically

2. Click **"Update Selected Agent"** to apply your selection

### 4. Testing Agents

Use the **🧪 Test Agent Responses** section to test different agents:

1. Enter a test message in the text input
2. Select an agent or provider from the dropdown
3. Click **"🧪 Test Response"** to see the response
4. Compare responses between different agents

## Monitoring Agent Usage

### Dashboard Overview

The main dashboard shows agent statistics:

- **Total Agents**: Number of available agents
- **Default Agent**: Currently set default agent
- **Agent Queries**: Total queries processed by agents
- **Agent Status**: Whether agents are enabled or disabled

### Usage Charts

- **Agent Usage Distribution**: Bar chart showing how many times each agent has been used
- **Agent Details**: Expandable section with detailed information about each agent

### Agent Details

For each agent, you can view:
- **Name and Description**
- **Type and Capabilities** 
- **Usage Count**: How many times the agent has been used
- **Error Count**: Number of errors encountered
- **Last Used**: When the agent was last used

## Agent vs Direct AI Provider

### When to Use Agents
- ✅ When you need specialized processing (document search, relationship analysis)
- ✅ When you want consistent behavior for specific types of queries
- ✅ When you need additional capabilities beyond basic chat

### When to Use Direct AI Provider
- ✅ For general conversation and chat
- ✅ When you want the fastest response time
- ✅ For simple questions that don't require specialized processing

## Testing the System

### Using the Test Script

Run the included test script to verify everything is working:

```bash
python test_agents.py
```

This will:
- Test agent initialization
- List available agents
- Test responses from each agent
- Show usage statistics
- Verify AI service integration

### Expected Output

You should see:
- ✅ Agent manager imported successfully
- 📋 List of available agents (RAG Bot, Graph RAG Bot)
- 🧪 Test responses from each agent
- 📊 Usage statistics
- ✅ All tests passed

## Troubleshooting

### No Agents Available
- Check that the agents module is properly installed
- Verify that agent_manager.py is in the agents/ directory
- Check the console for import errors

### Agent Responses Not Working
- Ensure agents are enabled in the settings
- Check that the selected agent is properly set
- Look for error messages in the Streamlit dashboard

### Performance Issues
- Agents add some processing overhead compared to direct AI providers
- For fastest responses, use direct AI providers
- Consider the trade-off between functionality and speed

## Development

### Adding New Agents

To add a new agent:

1. Create a new class inheriting from `AbstractAgent`
2. Implement the required `initialize()` and `invoke()` methods
3. Register the agent in `agent_manager.py`
4. Update this guide with the new agent's capabilities

### Customizing Agent Behavior

- Modify the agent classes in the `agents/` directory
- Update agent descriptions and capabilities
- Adjust routing logic in the agent manager

## Support

If you encounter issues:
1. Check the Streamlit dashboard for error messages
2. Run the test script to verify functionality
3. Check the console logs for detailed error information
4. Ensure all dependencies are properly installed 