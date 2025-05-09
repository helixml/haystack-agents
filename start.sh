#!/bin/bash

# Check if hayhooks is installed
if ! command -v hayhooks &> /dev/null; then
    echo "Hayhooks is not installed. Installing..."
    pip install hayhooks
fi

# Undeploy existing pipeline if it exists
echo "Cleaning up any existing deployments..."
hayhooks pipeline undeploy gpt-4o-mini 2>/dev/null

# Deploy the OpenAI-compatible endpoint
echo "Deploying OpenAI-compatible chat completions API..."
hayhooks pipeline deploy-files -n gpt-4o-mini agent_deployment

# Show usage information
echo -e "\nAvailable endpoint:"
echo "- OpenAI-compatible API: http://localhost:1416/v1/chat/completions"
echo -e "\nExample usage:"
echo -e "\n1. Streaming example:"
echo "curl -X POST 'http://localhost:1416/v1/chat/completions' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\":\"gpt-4o-mini\", \"messages\":[{\"role\":\"user\", \"content\":\"How is the weather in London?\"}], \"stream\":true}'"

echo -e "\n2. Non-streaming example:"
echo "curl -X POST 'http://localhost:1416/v1/chat/completions' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\":\"gpt-4o-mini\", \"messages\":[{\"role\":\"user\", \"content\":\"How is the weather in London?\"}], \"stream\":false}'"

# Start the Hayhooks server in the foreground
echo -e "\nStarting Hayhooks server at http://localhost:1416 (Press Ctrl+C to stop)..."
hayhooks run 