"""
Node 3: LLM Re-Ranking
Uses Llama via Groq to score and rank the MLS candidates against the buyer profile.
Returns the top 5 matches with reasoning.
"""
import json
import re
import os
from pathlib import Path
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.llm_utils import get_llm, llm_invoke_with_retry

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rerank_prompt.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def llm_rerank_node(state: AgentState) -> AgentState:
    """
    Re-ranks MLS candidates using LLM judgment and returns top matches.
    Skips gracefully if no candidates exist (vague leads, impossible budgets, etc.)
    """
    mls_candidates = state.get("mls_candidates") or []
    buyer_profile = state.get("buyer_profile") or {}

    if not mls_candidates:
        return {"ranked_matches": []}

    # Truncate candidate list to 20 for token safety
    candidates_to_rank = mls_candidates[:20]

    # Strip PII before sending to LLM (double-check)
    safe_candidates = []
    for c in candidates_to_rank:
        safe_c = {k: v for k, v in c.items() if k not in ("owner_name", "owner_phone")}
        safe_candidates.append(safe_c)

    prompt_template = _load_prompt()
    prompt = (
        prompt_template
        .replace("{buyer_profile}", json.dumps(buyer_profile, indent=2))
        .replace("{mls_candidates}", json.dumps(safe_candidates, indent=2))
    )

    llm = get_llm(temperature=0)
    raw = llm_invoke_with_retry(llm, [HumanMessage(content=prompt)], context="llm_rerank")

    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        ranked = result.get("ranked_matches", [])
    except json.JSONDecodeError:
        # Try to extract JSON array directly
        try:
            match = re.search(r'\{.*"ranked_matches".*\}', raw, re.DOTALL)
            if match:
                ranked = json.loads(match.group())["ranked_matches"]
            else:
                ranked = []
        except Exception:
            ranked = []

    # Enrich ranked matches with full listing data for brief generation
    listing_lookup = {c["listing_id"]: c for c in safe_candidates}
    enriched = []
    for match in ranked:
        lid = match.get("listing_id")
        if lid and lid in listing_lookup:
            match["listing"] = listing_lookup[lid]
            enriched.append(match)

    return {"ranked_matches": enriched}
