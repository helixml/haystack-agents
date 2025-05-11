#!/bin/bash

# Check if hayhooks is installed
if ! command -v hayhooks &> /dev/null; then
    echo "Hayhooks is not installed. Installing..."
    pip install hayhooks
fi

# Function to clean up background processes
cleanup() {
    echo -e "\nShutting down..."
    if [ -n "$SERVER_PID" ] && ps -p $SERVER_PID > /dev/null; then
        echo "Stopping Hayhooks server (PID: $SERVER_PID)..."
        kill $SERVER_PID
    fi
    exit 0
}

# Set up trap for Ctrl-C
trap cleanup SIGINT SIGTERM

# Kill any existing hayhooks processes
echo "Checking for existing hayhooks processes..."
pkill -f "hayhooks run" || echo "No running hayhooks processes found."
sleep 2  # Give processes time to terminate

# Start the Hayhooks server in the background
echo "Starting Hayhooks server in background..."
hayhooks run --host 0.0.0.0 > hayhooks.log 2>&1 &
SERVER_PID=$!

# Function to check if server is ready
wait_for_server() {
    echo "Waiting for Hayhooks server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:1416/docs > /dev/null; then
            echo "Server is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo "Server failed to start within 30 seconds"
    return 1
}

# Wait for server to be ready
if ! wait_for_server; then
    echo "Failed to start server. Check hayhooks.log for details."
    cleanup
    exit 1
fi

# Undeploy existing pipeline if it exists
echo "Cleaning up any existing deployments..."
hayhooks pipeline undeploy gpt-4o-mini || echo "No existing deployment found or failed to undeploy"

# Deploy the OpenAI-compatible endpoint
echo "Deploying OpenAI-compatible chat completions API..."
if ! hayhooks pipeline deploy-files -n gpt-4o-mini agent_deployment; then
    echo "Failed to deploy pipeline. Check hayhooks.log for details."
    cleanup
    exit 1
fi

echo -e "\nDeployment successful!"
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

# Show the server logs
echo -e "\nServer is running. Showing logs (Ctrl+C to stop)..."
tail -f hayhooks.log