"""
LangGraph state schema for the Buyer Lead Intake Agent.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph pipeline."""

    # Raw input
    lead: dict                        # Original lead dict from JSON

    # Node 1 output: parsed buyer profile
    buyer_profile: Optional[dict]

    # Node 1b output: anomaly / security flags
    anomaly_flags: list               # List of flag strings

    # Node 2 output: MLS candidates after rule-based pre-filter
    mls_candidates: list              # List of MLS row dicts (up to 20)

    # Node 3 output: LLM-ranked top matches
    ranked_matches: list              # List of {listing: dict, score: int, reasoning: str}

    # Node 4 output: final Markdown brief
    lead_brief_md: str
