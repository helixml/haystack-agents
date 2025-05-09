from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.websearch import SerperDevWebSearch
from haystack.tools.component_tool import ComponentTool
from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator
from typing import Any, Dict, List, Optional
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

class PipelineWrapper(BasePipelineWrapper):
    # Set name attribute to match expected model in requests
    name = "gpt-4o-mini"
    
    def setup(self) -> None:
        # Create the web search component
        web_search = SerperDevWebSearch(top_k=3)

        # Create the ComponentTool with simpler parameters
        web_tool = ComponentTool(
            component=web_search,
            name="web_search",
            description="Search the web for current information like weather, news, or facts."
        )

        # Create the agent with the web tool
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="""You're a helpful agent. When asked about current information like weather, news, or facts, 
                          use the web_search tool to find the information and then summarize the findings.
                          When you get web search results, extract the relevant information and present it in a clear, 
                          concise manner.""",
            tools=[web_tool]
        )

        # Create the pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("agent", agent)

    def run_api(self, query: str) -> str:
        """
        Standard API endpoint for the agent
        """
        user_message = ChatMessage.from_user(query)
        result = self.pipeline.run({"agent": {"messages": [user_message]}})
        return result["agent"]["messages"][-1].text

    def run_chat_completion(self, model: str, messages: List[Dict[str, str]], body: Dict[str, Any]) -> Any:
        """
        OpenAI-compatible chat completion API with streaming support
        """
        # Log the request body for debugging
        logger.info(f"Chat completion request body: {body}")
        
        question = get_last_user_message(messages)
        user_message = ChatMessage.from_user(question)
        
        # Check if streaming is requested - handle various formats
        # The stream parameter can be a boolean, string "true"/"false", or other format
        stream_param = body.get("stream", False)
        
        # Convert to boolean, handling string representations
        if isinstance(stream_param, str):
            stream = stream_param.lower() == "true"
        else:
            stream = bool(stream_param)
            
        logger.info(f"Stream parameter: {stream_param}, interpreted as: {stream}")
        
        if stream:
            # Stream the response to the client
            logger.info("Using streaming response")
            return streaming_generator(
                pipeline=self.pipeline, 
                pipeline_run_args={"agent": {"messages": [user_message]}}
            )
        else:
            # Return a complete response without streaming
            logger.info("Using non-streaming response")
            result = self.pipeline.run({"agent": {"messages": [user_message]}})
            response_text = result["agent"]["messages"][-1].text
            
            # Just return the text directly - Hayhooks will format it properly
            logger.info(f"Non-streaming response text: {response_text}")
            return response_text 