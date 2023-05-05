# Bring in deps
import os 

import streamlit as st 

from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain import SerpAPIWrapper
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish

def random_word(query: str) -> str:
    print("\\nNow I'm doing this!")
    return "foo"

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="RandomWord",
        func=random_word,
        description="call this to get a random word."
    )
]

class FakeAgent(BaseMultiActionAgent):
    """Fake Custom Agent."""
    @property
    def input_keys(self):
        return ["input"]

    def aplan(  # Rename this method to 'aplan'
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool="Search", tool_input=kwargs["input"], log=""),
                AgentAction(tool="RandomWord", tool_input=kwargs["input"], log=""),
            ]
        else:
            return AgentFinish(return_values={"output": "bar"}, log="")

    
agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def run_agent(parameter):
    agent_executor.run(parameter)

def main():
    st.title("Tech White Paper - GPT")
    
    parameter = st.text_input("Enter the topic that you'd like the technical white paper about...")
    
    if parameter:
        st.write(run_agent(parameter))
        st.write(f"Agent has run with the parameter: {parameter}")
        
    
