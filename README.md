# Haystack Agent with HTTP API via Hayhooks

This project demonstrates how to expose a Haystack Agent through HTTP endpoints using Hayhooks, allowing for both standard API calls and OpenAI-compatible streaming responses.

## Components

- **agent.py**: Contains the main implementation of a Haystack Agent with SerperDevWebSearch tool
- **agent_deployment**: Directory containing deployment files for Hayhooks:
  - **pipeline_wrapper.py**: Implementation of BasePipelineWrapper for standard API endpoints
  - **model_wrapper.py**: Implementation for OpenAI-compatible API endpoints

## API Endpoints

The agent is accessible through two types of API endpoints:

### Standard API Endpoint

```bash
curl -X POST "http://localhost:1416/agent_service/run" \
     -H "Content-Type: application/json" \
     -d '{"query":"How is the weather in London?"}'
```

### OpenAI-compatible Streaming API Endpoint

```bash
curl -X POST "http://localhost:1416/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o-mini",
       "messages": [{"role": "user", "content": "How is the weather in London?"}],
       "stream": true
     }'
```

## Implementation Details

1. **Standard Pipeline Wrapper**:
   - Initializes a Haystack Agent with a web search tool
   - Implements the `run_api` method for handling standard requests
   - Returns text responses from the agent

2. **OpenAI-compatible API**:
   - Uses the same agent pipeline
   - Implements the `run_chat_completion` method required by Hayhooks
   - Uses `streaming_generator` for streaming responses in OpenAI format
   - Sets the `name` attribute to match the expected model name

## Key Lessons

1. When implementing OpenAI compatibility, set the pipeline name to match the expected model name.
2. Use `streaming_generator` for streaming responses.
3. Implement both `run_api` and `run_chat_completion` methods for full compatibility.
4. Hayhooks serves on a default port of 1416. 