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

# Set up logging with more detail
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig()
logger = logging.getLogger(__name__)

class PipelineWrapper(BasePipelineWrapper):
    # Set name attribute to match expected model in requests
    name = "gpt-4o-mini"
    
    def setup(self) -> None:
        # Create the web search component
        logger.info("Setting up web search component...")
        web_search = SerperDevWebSearch(top_k=3)

        # Create the ComponentTool with simpler parameters
        logger.info("Creating web search tool...")
        web_tool = ComponentTool(
            component=web_search,
            name="web_search",
            description="Search the web for current information like weather, news, or facts."
        )

        # Create the agent with the web tool
        logger.info("Creating agent with OpenAI chat generator...")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            # TODO: instrument the actual Agent implementation instead of hacking around this with prompting
            system_prompt="""You are a helpful agent.
Your primary goal is to answer the user's question.
You have a tool called 'web_search' that can search the web for current information. Its description is: "Search the web for current information like weather, news, or facts."

When you decide to use the 'web_search' tool:
1. First, output an XML tag like this, containing the exact query you will use for the search: <tool_call name="web_search"><input>YOUR_SEARCH_QUERY</input></tool_call>
2. Then, make the actual call to the 'web_search' tool. (This happens automatically when you signal the tool call to the system).

After the 'web_search' tool has been executed and you receive its results from the system:
1. First, output an XML tag like this, summarizing or stating the raw output from the tool: <tool_result name="web_search"><output>TOOL_OUTPUT_SUMMARY_OR_DATA</output></tool_result>
2. Then, use this information to formulate your final answer to the user.

Example interaction if you need to search:
User: What's the weather in Paris?
Assistant: <tool_call name="web_search"><input>weather in Paris</input></tool_call>
(System executes web_search tool with "weather in Paris" and provides you the results, e.g., "Paris is sunny, 25°C")
Assistant: <tool_result name="web_search"><output>Paris is sunny, 25°C</output></tool_result>
The weather in Paris is sunny and 25°C.

If you don't need to use a tool, just answer directly.
When asked about current information (like weather, news, or general facts), you should use the 'web_search' tool.
After receiving search results, extract the relevant information and present it to the user in a clear, concise manner, following the XML formatting instructions above.""",
            tools=[web_tool]
        )

        # Create the pipeline
        logger.info("Creating pipeline...")
        self.pipeline = Pipeline()
        self.pipeline.add_component("agent", agent)
        logger.info("Pipeline setup complete!")

    def run_api(self, query: str) -> str:
        """
        Standard API endpoint for the agent
        """
        logger.info(f"Running API query: {query}")
        user_message = ChatMessage.from_user(query)
        result = self.pipeline.run({"agent": {"messages": [user_message]}})
        logger.info(f"API result: {result}")
        return result["agent"]["messages"][-1].text

    def run_chat_completion(self, model: str, messages: List[Dict[str, str]], body: Dict[str, Any]) -> Any:
        """
        OpenAI-compatible chat completion API with streaming support
        """
        # Log the request body for debugging
        logger.info(f"Chat completion request body: {body}")
        
        question = get_last_user_message(messages)
        logger.info(f"Extracted question: {question}")
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
            
            # Log the complete conversation history
            logger.info("Complete conversation history:")
            for msg in result["agent"]["messages"]:
                logger.info(f"{msg.role}: {msg.text}")
                if hasattr(msg, 'tool_call') and msg.tool_call:
                    logger.info(f"Tool call: {msg.tool_call}")
                if hasattr(msg, 'tool_call_result') and msg.tool_call_result:
                    logger.info(f"Tool result: {msg.tool_call_result}")
            
            logger.info(f"Final response text: {response_text}")
            return response_text 