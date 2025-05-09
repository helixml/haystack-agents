# Implementation Notes: Exposing Haystack Agents via HTTP with Hayhooks

## Key Findings

1. **Hayhooks Architecture**:
   - Hayhooks provides a framework for exposing Haystack pipelines via HTTP endpoints
   - Requires a `PipelineWrapper` class that extends `BasePipelineWrapper`
   - The wrapper needs to implement at least `setup()` and `run_api()` methods
   - OpenAI compatibility requires implementing the `run_chat_completion()` method

2. **OpenAI Compatibility**:
   - To expose a pipeline as an OpenAI-compatible endpoint, pipeline name must match the model name expected in requests
   - For streaming, use the `streaming_generator()` helper function provided by Hayhooks
   - OpenAI endpoints follow the standard OpenAI API format:
     - `/v1/chat/completions`
     - Accepts model, messages, and stream parameters

3. **Naming Conventions**:
   - For OpenAI compatibility, deploy a pipeline with a name matching the expected model name (e.g., `gpt-4o-mini`)
   - For standard API endpoints, any descriptive name works (e.g., `agent_service`)

4. **Multiple Deployments**:
   - The same pipeline implementation can be deployed multiple times with different names
   - This allows exposing the same functionality through both standard and OpenAI-compatible endpoints

5. **Port Configuration**:
   - Hayhooks defaults to port 1416, not 8000 as commonly used by other web frameworks
   - This can be configured via `HAYHOOKS_PORT` environment variable if needed

## Lessons Learned

1. **Pipeline Implementation**:
   - Creating the pipeline programmatically in `setup()` is more reliable than loading from YAML
   - Component imports must be fully qualified when using YAML (e.g., `haystack.components.agents.Agent`)

2. **Error Handling**:
   - When errors occur, check the Hayhooks server logs for detailed information
   - Use `hayhooks status` to verify which pipelines are currently deployed

3. **Streaming Implementation**:
   - The `streaming_generator()` function works well with Haystack pipelines that don't natively support streaming
   - The generator chunks the response automatically in the format expected by OpenAI clients

4. **Development Workflow**:
   - Use `hayhooks pipeline undeploy <name>` before redeploying to avoid conflicts
   - Use the `--overwrite` flag when making frequent changes during development

5. **Testing**:
   - Test both standard and OpenAI-compatible endpoints separately
   - When testing streaming, ensure the client can handle Server-Sent Events (SSE) format 