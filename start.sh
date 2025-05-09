#!/bin/bash

# Check if hayhooks is installed
if ! command -v hayhooks &> /dev/null; then
    echo "Hayhooks is not installed. Installing..."
    pip install hayhooks
fi

# Undeploy existing pipelines if they exist
echo "Cleaning up any existing deployments..."
hayhooks pipeline undeploy agent_service 2>/dev/null
hayhooks pipeline undeploy gpt-4o-mini 2>/dev/null

# Deploy the pipelines
echo "Deploying agent endpoints..."
hayhooks pipeline deploy-files -n agent_service agent_deployment
hayhooks pipeline deploy-files -n gpt-4o-mini agent_deployment

# Show available endpoints
echo -e "\nAvailable endpoints:"
echo "- Standard API: http://localhost:1416/agent_service/run"
echo "- OpenAI-compatible API: http://localhost:1416/v1/chat/completions"
echo -e "\nExample usage:"
echo "curl -X POST 'http://localhost:1416/agent_service/run' -H 'Content-Type: application/json' -d '{\"query\":\"How is the weather in London?\"}'"
echo "curl -X POST 'http://localhost:1416/v1/chat/completions' -H 'Content-Type: application/json' -d '{\"model\":\"gpt-4o-mini\", \"messages\":[{\"role\":\"user\", \"content\":\"How is the weather in London?\"}], \"stream\":true}'"
echo -e "\nHayhooks is now running. Press Ctrl+C to stop."

# In a real production environment, you would run the hayhooks server here
# But since we already have it running for this tutorial, we'll just wait for user input
read -p "Press Enter to exit..." 