"""Graph wiring.

    pre_guardrail ──► retrieve ──► prompt_builder ──► generate ──► post_guardrail ──► write
         │               │                               │                             ▲
         └─ blocked ─────┴─ off-topic ───────────────────┴─ error ─────────────────────┘

Every path ends in `write`, so every turn — including refusals and failures —
is persisted with its tokens, prompts and latencies.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from journey_agent.graph.nodes import AgentDeps, make_nodes
from journey_agent.graph.state import AgentState

PIPELINE = ["pre_guardrail", "retrieve", "prompt_builder", "generate", "post_guardrail", "write"]


def _continue_or_write(next_node: str):  # type: ignore[no-untyped-def]
    def route(state: AgentState) -> str:
        return next_node if state.get("route") == "continue" else "write"

    return route


def build_graph(deps: AgentDeps) -> CompiledStateGraph:  # type: ignore[type-arg]
    nodes = make_nodes(deps)
    graph = StateGraph(AgentState)
    for name in PIPELINE:
        graph.add_node(name, nodes[name])

    graph.add_edge(START, "pre_guardrail")
    graph.add_conditional_edges(
        "pre_guardrail", _continue_or_write("retrieve"), ["retrieve", "write"]
    )
    graph.add_conditional_edges(
        "retrieve", _continue_or_write("prompt_builder"), ["prompt_builder", "write"]
    )
    graph.add_edge("prompt_builder", "generate")
    graph.add_conditional_edges(
        "generate", _continue_or_write("post_guardrail"), ["post_guardrail", "write"]
    )
    graph.add_edge("post_guardrail", "write")
    graph.add_edge("write", END)
    return graph.compile()
