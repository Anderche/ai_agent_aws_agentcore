from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_aws import ChatBedrock
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import load_settings
from .tools import TOOLS


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


_MEMORY_STORE = None


def _resolve_memory_store(settings):
    if settings.memory_id:
        try:
            from langgraph.checkpoint.aws import AgentCoreMemoryStore

            return AgentCoreMemoryStore(
                memory_id=settings.memory_id,
                region_name=settings.aws_region,
            )
        except Exception:  # noqa: BLE001
            pass
    return None


def _resolve_checkpointer(settings):
    if settings.memory_id:
        try:
            from langgraph.checkpoint.aws import AgentCoreMemorySaver

            return AgentCoreMemorySaver(
                memory_id=settings.memory_id,
                region_name=settings.aws_region,
            )
        except Exception:  # noqa: BLE001
            pass

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def build_graph(settings=None):
    if settings is None:
        settings = load_settings()
    global _MEMORY_STORE  # noqa: PLW0603

    llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    graph_builder = StateGraph(AgentState)

    def chatbot(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=TOOLS))
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    checkpointer = _resolve_checkpointer(settings)
    store = _resolve_memory_store(settings)
    _MEMORY_STORE = store
    return graph_builder.compile(checkpointer=checkpointer, store=store)


def get_memory_store():
    return _MEMORY_STORE

