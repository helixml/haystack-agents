# Haystack Agent with OpenAI-Compatible API via Hayhooks

This project demonstrates how to expose a Haystack Agent through an OpenAI-compatible API using Hayhooks.

## Components

- **agent_deployment**: Directory containing deployment files for Hayhooks:
  - **pipeline_wrapper.py**: Implements the OpenAI-compatible API with both streaming and non-streaming support

## API Endpoints

The agent is accessible through an OpenAI-compatible chat completions API:

### Streaming Example

```bash
curl -X POST "http://localhost:1416/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
         "model": "gpt-4o-mini",
         "messages": [{"role": "user", "content": "How is the weather in London?"}],
         "stream": true
     }'
```

### Non-Streaming Example

```bash
curl -X POST "http://localhost:1416/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
         "model": "gpt-4o-mini",
         "messages": [{"role": "user", "content": "How is the weather in London?"}],
         "stream": false
     }'
```

## Setup and Usage

1. Make sure you have Haystack and Hayhooks installed:
   ```
   pip install haystack-ai hayhooks
   ```

2. Run the start script to deploy the agent:
   ```
   ./start.sh
   ```

3. In a separate terminal, send requests to the API endpoint as shown above.

## Implementation Details

The implementation:
- Uses Haystack's Agent with the SerperDevWebSearch tool
- Exposes the agent through Hayhooks' OpenAI-compatible interface
- Supports both streaming and non-streaming responses
- Deploys the pipeline with name 'gpt-4o-mini' which is used as the model name in API requests

## Key Points

1. Hayhooks requires a file named `pipeline_wrapper.py` that implements the `BasePipelineWrapper` interface
2. The pipeline name must match the model name in OpenAI API requests
3. The wrapper class must implement `run_chat_completion` method for OpenAI compatibility
4. Streaming responses are handled by the `streaming_generator` helper provided by Hayhooks
5. Non-streaming responses are formatted to match the OpenAI API response format 