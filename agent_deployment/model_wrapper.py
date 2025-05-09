from pathlib import Path
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.websearch import SerperDevWebSearch
from haystack.tools.component_tool import ComponentTool
from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator

class PipelineWrapper(BasePipelineWrapper):
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

    def run_chat_completion(self, model: str, messages: list, body: dict):
        """
        OpenAI-compatible chat completion API with streaming support
        """
        question = get_last_user_message(messages)
        user_message = ChatMessage.from_user(question)
        
        # Stream the response to the client
        return streaming_generator(
            pipeline=self.pipeline, 
            pipeline_run_args={"agent": {"messages": [user_message]}}
        ) 