from haystack import Pipeline
from haystack.dataclasses import ChatMessage
# from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.websearch import SerperDevWebSearch
from haystack.tools.component_tool import ComponentTool
from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator
from typing import Any, Dict, List, Optional, Iterator, Union
import time
import logging
import re
import json
from dataclasses import dataclass, field
from threading import Lock
import queue
import threading

# Set up logging with more detail
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig()
logger = logging.getLogger(__name__)


# Create a shared mutable class to pass information between Agent and streaming wrapper
class SharedToolInfo:
    def __init__(self):
        self.lock = Lock()
        self.pending_tool_calls = []
        self.pending_tool_results = []
        # Create a queue for the merged output stream
        self.output_queue = queue.Queue()
        # Flag to signal when the original generator is done
        self.generator_done = threading.Event()
        logger.debug("APPLE: SharedToolInfo initialized")
        
    def add_tool_call(self, tool_name, query):
        with self.lock:
            tool_call_info = {
                "tool_name": tool_name,
                "query": query,
                "inserted": False
            }
            self.pending_tool_calls.append(tool_call_info)
            logger.debug(f"APPLE: Added tool call to shared info: {tool_name}, query: {query}")
            logger.debug(f"APPLE: Current pending tool calls: {len(self.pending_tool_calls)}")
            
            # Directly put tool call XML into the output queue
            xml = f"<tool_call name=\"{tool_name}\"><input>{query}</input></tool_call>\n"
            self.output_queue.put(xml)
            logger.debug(f"APPLE: Directly added tool call XML to output queue")
            
    def add_tool_result(self, tool_name, result):
        with self.lock:
            tool_result_info = {
                "tool_name": tool_name,
                "result": result,
                "inserted": False
            }
            self.pending_tool_results.append(tool_result_info)
            logger.debug(f"APPLE: Added tool result to shared info: {tool_name}, result (truncated): {result[:100]}...")
            logger.debug(f"APPLE: Current pending tool results: {len(self.pending_tool_results)}")
            
            # Directly put tool result XML into the output queue
            xml = f"<tool_result name=\"{tool_name}\"><output>{result}</output></tool_result>\n"
            self.output_queue.put(xml)
            logger.debug(f"APPLE: Directly added tool result XML to output queue")
            
    def get_next_uninserted_tool_call(self):
        with self.lock:
            logger.debug(f"APPLE: Checking for uninserted tool calls, total: {len(self.pending_tool_calls)}")
            for i, item in enumerate(self.pending_tool_calls):
                if not item["inserted"]:
                    item["inserted"] = True
                    logger.debug(f"APPLE: Found uninserted tool call #{i}: {item['tool_name']}, query: {item['query']}")
                    return item
            logger.debug("APPLE: No uninserted tool calls found")
            return None
            
    def get_next_uninserted_tool_result(self):
        with self.lock:
            logger.debug(f"APPLE: Checking for uninserted tool results, total: {len(self.pending_tool_results)}")
            for i, item in enumerate(self.pending_tool_results):
                if not item["inserted"]:
                    item["inserted"] = True
                    logger.debug(f"APPLE: Found uninserted tool result #{i}: {item['tool_name']}")
                    return item
            logger.debug("APPLE: No uninserted tool results found")
            return None
    
    def reset(self):
        """Reset the state for a new streaming session"""
        with self.lock:
            self.pending_tool_calls = []
            self.pending_tool_results = []
            # Clear any items in the output queue
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            # Reset the done flag
            self.generator_done.clear()
            logger.debug("APPLE: SharedToolInfo reset for new streaming session")

# Create a shared instance
shared_tool_info = SharedToolInfo()


# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging, tracing
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.pipeline import Pipeline
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.state import State, _schema_from_dict, _schema_to_dict, _validate_schema
from haystack.dataclasses.state_utils import merge_lists
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.tools import Tool, Toolset, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)


@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until a exit_condition condition is met.
    The exit_condition can be triggered either by a direct text response or by invoking a specific designated tool.

    When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

    ### Usage example
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools.tool import Tool

    tools = [Tool(name="calculator", description="..."), Tool(name="search", description="...")]

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=tools,
        exit_condition="search",
    )

    # Run the agent
    result = agent.run(
        messages=[ChatMessage.from_user("Find information about Haystack")]
    )

    assert "messages" in result  # Contains conversation history
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[List[str]] = None,
        state_schema: Optional[Dict[str, Any]] = None,
        max_agent_steps: int = 100,
        raise_on_tool_invocation_failure: bool = False,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ):
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: List of Tool objects or a Toolset that the agent can use.
        :param system_prompt: System prompt for the agent.
        :param exit_conditions: List of conditions that will cause the agent to return.
            Can include "text" if the agent should return when it generates a message without tool calls,
            or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_agent_steps: Maximum number of steps the agent will run before stopping. Defaults to 100.
            If the agent exceeds this number of steps, it will stop and return the current state.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
        """
        # Check if chat_generator supports tools parameter
        chat_generator_run_method = inspect.signature(chat_generator.run)
        if "tools" not in chat_generator_run_method.parameters:
            raise TypeError(
                f"{type(chat_generator).__name__} does not accept tools parameter in its run method. "
                "The Agent component requires a chat generator that supports tools."
            )

        valid_exits = ["text"] + [tool.name for tool in tools or []]
        if exit_conditions is None:
            exit_conditions = ["text"]
        if not all(condition in valid_exits for condition in exit_conditions):
            raise ValueError(
                f"Invalid exit conditions provided: {exit_conditions}. "
                f"Valid exit conditions must be a subset of {valid_exits}. "
                "Ensure that each exit condition corresponds to either 'text' or a valid tool name."
            )

        # Validate state schema if provided
        if state_schema is not None:
            _validate_schema(state_schema)
        self._state_schema = state_schema or {}

        # Initialize state schema
        resolved_state_schema = _deepcopy_with_exceptions(self._state_schema)
        if resolved_state_schema.get("messages") is None:
            resolved_state_schema["messages"] = {"type": List[ChatMessage], "handler": merge_lists}
        self.state_schema = resolved_state_schema

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        output_types = {}
        for param, config in self.state_schema.items():
            output_types[param] = config["type"]
            # Skip setting input types for parameters that are already in the run method
            if param in ["messages", "streaming_callback"]:
                continue
            component.set_input_type(self, name=param, type=config["type"], default=None)
        component.set_output_types(self, **output_types)

        self._tool_invoker = None
        if self.tools:
            self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=self.raise_on_tool_invocation_failure)
        else:
            logger.warning(
                "No tools provided to the Agent. The Agent will behave like a ChatGenerator and only return text "
                "responses. To enable tool usage, pass tools directly to the Agent, not to the chat_generator."
            )

        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the Agent.
        """
        if not self._is_warmed_up:
            if hasattr(self.chat_generator, "warm_up"):
                self.chat_generator.warm_up()
            self._is_warmed_up = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        if self.streaming_callback is not None:
            streaming_callback = serialize_callable(self.streaming_callback)
        else:
            streaming_callback = None

        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            tools=serialize_tools_or_toolset(self.tools),
            system_prompt=self.system_prompt,
            exit_conditions=self.exit_conditions,
            # We serialize the original state schema, not the resolved one to reflect the original user input
            state_schema=_schema_to_dict(self._state_schema),
            max_agent_steps=self.max_agent_steps,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            streaming_callback=streaming_callback,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """
        Deserialize the agent from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized agent
        """
        init_params = data.get("init_parameters", {})

        deserialize_chatgenerator_inplace(init_params, key="chat_generator")

        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_or_toolset_inplace(init_params, key="tools")

        return default_from_dict(cls, data)

    def _prepare_generator_inputs(self, streaming_callback: Optional[StreamingCallbackT] = None) -> Dict[str, Any]:
        """Prepare inputs for the chat generator."""
        generator_inputs: Dict[str, Any] = {"tools": self.tools}
        selected_callback = streaming_callback or self.streaming_callback
        if selected_callback is not None:
            generator_inputs["streaming_callback"] = selected_callback
        return generator_inputs

    def _create_agent_span(self) -> Any:
        """Create a span for the agent run."""
        return tracing.tracer.trace(
            "haystack.agent.run",
            tags={
                "haystack.agent.max_steps": self.max_agent_steps,
                "haystack.agent.tools": self.tools,
                "haystack.agent.exit_conditions": self.exit_conditions,
                "haystack.agent.state_schema": _schema_to_dict(self.state_schema),
            },
        )

    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process messages and execute tools until the exit condition is met.

        :param messages: List of chat messages to process
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        logger.debug("Agent.run() method called")
        if not self._is_warmed_up and hasattr(self.chat_generator, "warm_up"):
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run()'.")

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages

        state = State(schema=self.state_schema, data=kwargs)
        state.set("messages", messages)

        generator_inputs = self._prepare_generator_inputs(streaming_callback=streaming_callback)

        component_visits = dict.fromkeys(["chat_generator", "tool_invoker"], 0)
        with self._create_agent_span() as span:
            span.set_content_tag(
                "haystack.agent.input",
                _deepcopy_with_exceptions({"messages": messages, "streaming_callback": streaming_callback, **kwargs}),
            )
            counter = 0
            while counter < self.max_agent_steps:
                # 1. Call the ChatGenerator
                llm_messages = Pipeline._run_component(
                    component_name="chat_generator",
                    component={"instance": self.chat_generator},
                    inputs={"messages": messages, **generator_inputs},
                    component_visits=component_visits,
                    parent_span=span,
                )["replies"]
                
                # Log tool calls for debugging
                for msg in llm_messages:
                    if hasattr(msg, 'tool_call') and msg.tool_call:
                        logger.debug(f"Tool call detected: {msg.tool_call}")
                        # Write tool call to shared info for streaming
                        if hasattr(msg.tool_call, 'tool_name') and msg.tool_call.tool_name:
                            # Try to extract query from arguments or input
                            query = ""
                            if hasattr(msg.tool_call, 'arguments') and isinstance(msg.tool_call.arguments, dict) and 'query' in msg.tool_call.arguments:
                                query = msg.tool_call.arguments['query']
                            elif hasattr(msg.tool_call, 'input') and isinstance(msg.tool_call.input, dict) and 'query' in msg.tool_call.input:
                                query = msg.tool_call.input['query']
                            elif hasattr(msg.tool_call, 'arguments') and isinstance(msg.tool_call.arguments, str):
                                try:
                                    args = json.loads(msg.tool_call.arguments)
                                    if isinstance(args, dict) and 'query' in args:
                                        query = args['query']
                                except:
                                    pass
                            
                            # If we still don't have a query, use the tool name as fallback
                            if not query:
                                query = str(msg.tool_call.tool_name)
                                
                            logger.debug(f"FISH: Adding tool call to shared info: {msg.tool_call.tool_name}, {query}")
                            shared_tool_info.add_tool_call(msg.tool_call.tool_name, query)
                
                state.set("messages", llm_messages)

                # 2. Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    counter += 1
                    break

                # 3. Call the ToolInvoker
                # We only send the messages from the LLM to the tool invoker
                tool_invoker_result = Pipeline._run_component(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    inputs={"messages": llm_messages, "state": state},
                    component_visits=component_visits,
                    parent_span=span,
                )
                tool_messages = tool_invoker_result["tool_messages"]
                
                # Log tool results for debugging
                for msg in tool_messages:
                    if hasattr(msg, 'tool_call_result') and msg.tool_call_result:
                        logger.debug(f"Tool result: {msg.tool_call_result}")
                        # Write tool result to shared info for streaming
                        if hasattr(msg.tool_call_result, 'origin') and hasattr(msg.tool_call_result.origin, 'tool_name'):
                            tool_name = msg.tool_call_result.origin.tool_name
                            result = ""
                            if hasattr(msg.tool_call_result, 'data'):
                                result = str(msg.tool_call_result.data)
                            elif hasattr(msg.tool_call_result, 'result'):
                                result = str(msg.tool_call_result.result)
                            else:
                                result = str(msg.tool_call_result)
                                
                            logger.debug(f"FISH: Adding tool result to shared info: {tool_name}, {result[:100]}...")
                            shared_tool_info.add_tool_result(tool_name, result)
                
                state = tool_invoker_result["state"]
                state.set("messages", tool_messages)

                # 4. Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    counter += 1
                    break

                # 5. Fetch the combined messages and send them back to the LLM
                messages = state.get("messages")
                counter += 1

            if counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", state.data)
            span.set_tag("haystack.agent.steps_taken", counter)
        return state.data

    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Asynchronously process messages and execute tools until the exit condition is met.

        This is the asynchronous version of the `run` method. It follows the same logic but uses
        asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
        if available.

        :param messages: List of chat messages to process
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        logger.debug("FISH Agent.run_async() method called")
        if not self._is_warmed_up and hasattr(self.chat_generator, "warm_up"):
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run_async()'.")

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages

        state = State(schema=self.state_schema, data=kwargs)
        state.set("messages", messages)

        generator_inputs = self._prepare_generator_inputs(streaming_callback=streaming_callback)

        component_visits = dict.fromkeys(["chat_generator", "tool_invoker"], 0)
        with self._create_agent_span() as span:
            span.set_content_tag(
                "haystack.agent.input",
                _deepcopy_with_exceptions({"messages": messages, "streaming_callback": streaming_callback, **kwargs}),
            )
            counter = 0
            while counter < self.max_agent_steps:
                # 1. Call the ChatGenerator
                result = await AsyncPipeline._run_component_async(
                    component_name="chat_generator",
                    component={"instance": self.chat_generator},
                    component_inputs={"messages": messages, **generator_inputs},
                    component_visits=component_visits,
                    max_runs_per_component=self.max_agent_steps,
                    parent_span=span,
                )
                llm_messages = result["replies"]
                state.set("messages", llm_messages)

                # 2. Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    counter += 1
                    break

                # 3. Call the ToolInvoker
                # We only send the messages from the LLM to the tool invoker
                # Check if the ToolInvoker supports async execution. Currently, it doesn't.
                tool_invoker_result = await AsyncPipeline._run_component_async(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    component_inputs={"messages": llm_messages, "state": state},
                    component_visits=component_visits,
                    max_runs_per_component=self.max_agent_steps,
                    parent_span=span,
                )
                tool_messages = tool_invoker_result["tool_messages"]
                state = tool_invoker_result["state"]
                state.set("messages", tool_messages)

                # 4. Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    counter += 1
                    break

                # 5. Fetch the combined messages and send them back to the LLM
                messages = state.get("messages")
                counter += 1

            if counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", state.data)
            span.set_tag("haystack.agent.steps_taken", counter)
        return state.data

    def _check_exit_conditions(self, llm_messages: List[ChatMessage], tool_messages: List[ChatMessage]) -> bool:
        """
        Check if any of the LLM messages' tool calls match an exit condition and if there are no errors.

        :param llm_messages: List of messages from the LLM
        :param tool_messages: List of messages from the tool invoker
        :return: True if an exit condition is met and there are no errors, False otherwise
        """
        matched_exit_conditions = set()
        has_errors = False

        for msg in llm_messages:
            if msg.tool_call and msg.tool_call.tool_name in self.exit_conditions:
                matched_exit_conditions.add(msg.tool_call.tool_name)

                # Check if any error is specifically from the tool matching the exit condition
                tool_errors = [
                    tool_msg.tool_call_result.error
                    for tool_msg in tool_messages
                    if tool_msg.tool_call_result is not None
                    and tool_msg.tool_call_result.origin.tool_name == msg.tool_call.tool_name
                ]
                if any(tool_errors):
                    has_errors = True
                    # No need to check further if we found an error
                    break

        # Only return True if at least one exit condition was matched AND none had errors
        return bool(matched_exit_conditions) and not has_errors



class PipelineWrapper(BasePipelineWrapper):
    # Set name attribute to match expected model in requests
    name = "gpt-4o-mini"
    
    # Add this function to help debug the streaming format
    def _dump_openai_tool_example(self):
        """
        Create a sample of what we expect OpenAI tool calls to look like in the streaming format.
        This is for debugging purposes.
        """
        example = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1694268190,
            "model": "gpt-4o-mini",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": "{\"query\":\"weather in London\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": None
                }
            ]
        }
        logger.debug(f"FISH EXAMPLE: Expected OpenAI format: {json.dumps(example, default=str)}")
        return example
    
    def setup(self) -> None:
        # Create the web search component
        logger.info("Setting up web search component...")
        web_search = SerperDevWebSearch(top_k=3)

        # TODO: add some more tools for a better demo...

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

When you need current information about the weather, news, facts, or other real-time data, use the 'web_search' tool to find accurate information.

After receiving search results, extract the relevant information and present it to the user in a clear, concise manner.

If you don't need to use a tool, just answer directly.""",
            tools=[web_tool]
        )

        # Create the pipeline
        logger.info("Creating pipeline...")
        self.pipeline = Pipeline()
        self.pipeline.add_component("agent", agent)
        logger.info("Pipeline setup complete!")
        
    def xml_streaming_wrapper(
        self, 
        original_generator: Iterator[Any]
    ) -> Iterator[Any]:
        """
        Wraps the original streaming generator to inject XML tags for tool calls and results.
        
        Uses a true interleaving approach with a background thread that processes the
        original LLM stream while tool calls and results are added directly to the output
        queue from their respective threads.
        
        This provides immediate feedback to the user when tools are called without any
        dependency on the timing of the LLM's responses.
        """
        logger.debug("APPLE STREAM: Starting XML streaming wrapper")
        
        # Reset shared info for this new run
        shared_tool_info.reset()
        logger.debug("APPLE STREAM: Reset shared tool info")
        
        # Function for background thread to process original generator
        def process_original_generator():
            try:
                chunk_count = 0
                first_chunk_logged = False
                
                for chunk in original_generator:
                    chunk_count += 1
                    
                    # Skip empty chunks
                    if not chunk:
                        logger.debug("APPLE STREAM: Empty chunk received")
                        continue
                    
                    # Log the original chunk format once
                    if not first_chunk_logged and chunk:
                        logger.debug(f"PEAR: ORIGINAL CHUNK FORMAT: {repr(chunk)}")
                        logger.debug(f"PEAR: ORIGINAL CHUNK TYPE: {type(chunk)}")
                        first_chunk_logged = True
                    
                    # Log and put the chunk in the output queue
                    logger.debug(f"APPLE STREAM: Putting original chunk {chunk_count} in output queue: {repr(chunk[:50])}...")
                    if isinstance(chunk, str) and chunk.strip():
                        logger.debug(f"PEAR: CONTENT IN CHUNK #{chunk_count}: {repr(chunk[:50])}...")
                    
                    # Put the chunk in the shared output queue
                    shared_tool_info.output_queue.put(chunk)
                
                logger.debug("APPLE STREAM: Original generator completed")
            except Exception as e:
                logger.error(f"APPLE STREAM: Error processing original generator: {e}")
            finally:
                # Signal that the generator is done
                shared_tool_info.generator_done.set()
                logger.debug("APPLE STREAM: Set generator_done flag")
        
        # Start the background thread to process the original generator
        thread = threading.Thread(target=process_original_generator)
        thread.daemon = True
        thread.start()
        logger.debug("APPLE STREAM: Started background thread for original generator")
        
        # Yield from the shared output queue until both sources are done
        while not shared_tool_info.generator_done.is_set() or not shared_tool_info.output_queue.empty():
            try:
                # Get with a timeout to allow checking if we're done
                item = shared_tool_info.output_queue.get(timeout=0.1)
                logger.debug(f"APPLE STREAM: Yielding from output queue: {repr(item[:50] if isinstance(item, str) else item)[:100]}...")
                yield item
            except queue.Empty:
                # Queue is empty but generator might not be done
                continue
        
        logger.debug("APPLE STREAM: Streaming complete")

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
            logger.info("Using streaming response with XML tags")
            # Output an example of what we expect for debugging
            self._dump_openai_tool_example()
            original_stream = streaming_generator(
                pipeline=self.pipeline, 
                pipeline_run_args={"agent": {"messages": [user_message]}}
            )
            return self.xml_streaming_wrapper(original_stream)
        else:
            # Return a complete response without streaming
            logger.info("Using non-streaming response")
            result = self.pipeline.run({"agent": {"messages": [user_message]}})
            
            # Begin constructing the full response with tool calls and results
            full_response = ""
            tool_call_found = False
            tool_result_found = False
            
            # Log the complete conversation history
            logger.info("Complete conversation history:")
            for idx, msg in enumerate(result["agent"]["messages"]):
                logger.info(f"Message #{idx} role: {msg.role}, text: {msg.text}")
                logger.info(f"FISH: Message #{idx} dir(msg): {dir(msg)}")
                logger.info(f"FISH: Message #{idx} type: {type(msg)}")
                
                # Let's find all attributes with values
                for attr in dir(msg):
                    if not attr.startswith('_') and not callable(getattr(msg, attr)):
                        try:
                            value = getattr(msg, attr)
                            logger.info(f"FISH: Message #{idx} {attr} = {value}")
                        except Exception as e:
                            logger.info(f"FISH: Error getting {attr}: {e}")
                
                # Process tool calls
                if hasattr(msg, 'tool_call') and msg.tool_call:
                    logger.info(f"FISH: Tool call object: {msg.tool_call}")
                    logger.info(f"FISH: Tool call type: {type(msg.tool_call)}")
                    logger.info(f"FISH: Tool call dir: {dir(msg.tool_call)}")
                    
                    # ToolCall is an object, not a dictionary, so access its properties directly
                    tool_name = msg.tool_call.tool_name if hasattr(msg.tool_call, 'tool_name') else "unknown_tool"
                    logger.info(f"FISH: Extracted tool name: {tool_name}")
                    
                    # The input structure depends on the tool implementation
                    try:
                        # Try accessing as an attribute first
                        if hasattr(msg.tool_call, 'input') and isinstance(msg.tool_call.input, dict):
                            tool_input = msg.tool_call.input
                            logger.info(f"FISH: Found tool input as dict attribute: {tool_input}")
                        # Try the arguments attribute which is already a dictionary in our case
                        elif hasattr(msg.tool_call, 'arguments') and isinstance(msg.tool_call.arguments, dict):
                            tool_input = msg.tool_call.arguments
                            logger.info(f"FISH: Found arguments as dict: {tool_input}")
                        # If it's a string, try parsing it as JSON
                        elif hasattr(msg.tool_call, 'arguments') and isinstance(msg.tool_call.arguments, str):
                            logger.info(f"FISH: Found arguments as string, parsing as JSON: {msg.tool_call.arguments}")
                            try:
                                tool_input = json.loads(msg.tool_call.arguments)
                                logger.info(f"FISH: Parsed arguments as JSON: {tool_input}")
                            except json.JSONDecodeError:
                                logger.info(f"FISH: Failed to parse arguments as JSON")
                                # Just use the user's question as the fallback
                                tool_input = {"query": question}
                                logger.info(f"FISH: Using user question as fallback: {question}")
                        # If none of those work, try direct access
                        elif hasattr(msg.tool_call, 'query'):
                            logger.info(f"FISH: Found query attribute: {msg.tool_call.query}")
                            tool_input = {"query": msg.tool_call.query}
                        else:
                            # If we can't find a query, use the raw properties
                            logger.info(f"FISH: No standard query attribute found, using str(msg.tool_call)")
                            tool_input = {"query": str(msg.tool_call)}
                    except Exception as e:
                        logger.error(f"FISH: Error extracting tool input: {e}")
                        # Just use the user's question as the fallback
                        tool_input = {"query": question}
                        logger.info(f"FISH: Using user question as fallback: {question}")
                    
                    # Check if we have a query to use
                    if isinstance(tool_input, dict) and "query" in tool_input:
                        query_value = tool_input["query"]
                        logger.info(f"FISH: Found query in tool_input: {query_value}")
                    else:
                        # Default to the user's question if we can't find a specific query
                        query_value = question
                        logger.info(f"FISH: Using original question as query: {query_value}")
                    
                    tool_call_found = True
                    full_response += f"<tool_call name=\"{tool_name}\"><input>{query_value}</input></tool_call>\n"
                    
                    # If this is the web_search tool, do some extra debug
                    if tool_name == "web_search":
                        logger.info(f"FISH: Special web_search debug:")
                        logger.info(f"FISH: Original question: {question}")
                        logger.info(f"FISH: Final query_value used: {query_value}")
                        # Try to extract the input in different ways
                        if hasattr(msg.tool_call, 'function'):
                            logger.info(f"FISH: web_search has function attribute: {msg.tool_call.function}")
                            if hasattr(msg.tool_call.function, 'arguments'):
                                logger.info(f"FISH: web_search function arguments: {msg.tool_call.function.arguments}")
                        if hasattr(msg.tool_call, 'params'):
                            logger.info(f"FISH: web_search has params: {msg.tool_call.params}")
                        if hasattr(msg.tool_call, 'input'):
                            logger.info(f"FISH: web_search has input: {msg.tool_call.input}")
                        logger.info(f"FISH: web_search tool_call str: {str(msg.tool_call)}")
                
                # Process tool results
                if hasattr(msg, 'tool_call_result') and msg.tool_call_result:
                    logger.info(f"Tool result: {msg.tool_call_result}")
                    tool_name = "unknown_tool"
                    result_value = ""
                    
                    # Extract tool name and result safely
                    if hasattr(msg.tool_call_result, 'origin') and hasattr(msg.tool_call_result.origin, 'tool_name'):
                        tool_name = msg.tool_call_result.origin.tool_name
                    
                    # Get result data
                    if hasattr(msg.tool_call_result, 'data'):
                        result_value = str(msg.tool_call_result.data)
                    elif hasattr(msg.tool_call_result, 'result'):
                        result_value = str(msg.tool_call_result.result)
                    else:
                        result_value = str(msg.tool_call_result)
                    
                    tool_result_found = True
                    full_response += f"<tool_result name=\"{tool_name}\"><output>{result_value}</output></tool_result>\n"
            
            # If we didn't find explicit tool calls/results but this is clearly a tool-using interaction,
            # try to extract them from the final text response
            if not (tool_call_found and tool_result_found) and result["agent"]["messages"][-1].text:
                # Just use the final text response without trying to guess about tools
                full_response = result["agent"]["messages"][-1].text
            else:
                # Add the final response text if present
                if result["agent"]["messages"][-1].role == "assistant" and result["agent"]["messages"][-1].text:
                    full_response += result["agent"]["messages"][-1].text
            
            logger.info(f"Final response with XML tags: {full_response}")
            return full_response.strip() 