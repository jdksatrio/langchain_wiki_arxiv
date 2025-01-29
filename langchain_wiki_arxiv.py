from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode,tools_condition
import os
from dotenv import load_dotenv
# Load api key from .env file
load_dotenv()


arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

tools = [wiki_tool, arxiv_tool]

class State(TypedDict):
  messages:Annotated[list,add_messages]

graph_builder= StateGraph(State)

llm=ChatGroq(model_name="Gemma2-9b-It")
bound_llm_tools = llm.bind_tools(tools=tools)

def chatbot(state:State):
  return {"messages":[bound_llm_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START,"chatbot")

graph=graph_builder.compile()

def stream_chat():
    user_input = input("Input: ")
    events=graph.stream(
     {"messages": [("user", user_input)]},stream_mode="values"
)

    for event in events:
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    stream_chat()