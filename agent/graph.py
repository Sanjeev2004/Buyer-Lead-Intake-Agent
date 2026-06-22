"""
LangGraph state machine for the Buyer Lead Intake Agent.
Defines the 5-node pipeline and wires them together.
"""
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.parse_classify import parse_classify_node
from agent.nodes.safety_guard import safety_guard_node
from agent.nodes.rule_filter import rule_filter_node
from agent.nodes.llm_rerank import llm_rerank_node
from agent.nodes.brief_generator import brief_generator_node


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph agent pipeline.

    Pipeline:
      parse_classify → safety_guard → rule_filter → llm_rerank → brief_generator → END
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("parse_classify", parse_classify_node)
    builder.add_node("safety_guard", safety_guard_node)
    builder.add_node("rule_filter", rule_filter_node)
    builder.add_node("llm_rerank", llm_rerank_node)
    builder.add_node("brief_generator", brief_generator_node)

    # Wire edges (linear pipeline)
    builder.set_entry_point("parse_classify")
    builder.add_edge("parse_classify", "safety_guard")
    builder.add_edge("safety_guard", "rule_filter")
    builder.add_edge("rule_filter", "llm_rerank")
    builder.add_edge("llm_rerank", "brief_generator")
    builder.add_edge("brief_generator", END)

    return builder.compile()
