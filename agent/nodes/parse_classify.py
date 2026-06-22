"""
Node 1: Parse & Classify
Uses Llama via Groq to extract a structured buyer profile from the raw inquiry message.
"""
import json
import re
import os
from pathlib import Path
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.llm_utils import get_llm, llm_invoke_with_retry

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "parse_prompt.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def parse_classify_node(state: AgentState) -> AgentState:
    """
    Extracts a structured buyer profile from the raw lead message using the LLM.
    """
    lead = state["lead"]
    prompt_template = _load_prompt()

    prompt = (
        prompt_template
        .replace("{buyer_name}", lead.get("buyer_name", "Unknown"))
        .replace("{channel}", lead.get("channel", "Unknown"))
        .replace("{message}", lead.get("message", ""))
    )

    llm = get_llm(temperature=0)
    raw = llm_invoke_with_retry(llm, [HumanMessage(content=prompt)], context="parse_classify")

    # Strip any markdown code fences if the model wraps JSON in them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        buyer_profile = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a minimal profile so the pipeline can continue
        buyer_profile = {
            "buyer_intent": "unknown",
            "property_types": ["any"],
            "bedrooms_min": None,
            "bedrooms_max": None,
            "budget_min": None,
            "budget_max": None,
            "neighborhoods": [],
            "required_features": [],
            "nice_to_have_features": [],
            "timeline": "unknown",
            "urgency": "medium",
            "buyer_type": "unknown",
            "is_cash_buyer": None,
            "special_needs": None,
            "notes": f"Profile extraction failed. Raw LLM output: {raw[:300]}",
        }

    return {
        "buyer_profile": buyer_profile,
        "anomaly_flags": [],
        "mls_candidates": [],
        "ranked_matches": [],
        "lead_brief_md": "",
    }
