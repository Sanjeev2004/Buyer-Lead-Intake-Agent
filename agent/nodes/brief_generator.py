"""
Node 4: Lead Brief Generator
Uses Llama via Groq to write the final rich Markdown Lead Brief.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.llm_utils import get_llm, llm_invoke_with_retry

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "brief_prompt.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _format_received_at(raw: str) -> str:
    """Pretty-print ISO datetime for the brief header."""
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except Exception:
        return raw


def brief_generator_node(state: AgentState) -> AgentState:
    """
    Generates the final Markdown Lead Brief using LLM.
    """
    lead = state["lead"]
    buyer_profile = state.get("buyer_profile") or {}
    anomaly_flags = state.get("anomaly_flags") or []
    ranked_matches = state.get("ranked_matches") or []

    # Strip PII from ranked matches before sending to LLM (triple safety check)
    safe_matches = []
    for m in ranked_matches:
        safe_m = dict(m)
        if "listing" in safe_m:
            safe_m["listing"] = {
                k: v for k, v in safe_m["listing"].items()
                if k not in ("owner_name", "owner_phone")
            }
        safe_matches.append(safe_m)

    prompt_template = _load_prompt()

    is_cash = buyer_profile.get("is_cash_buyer")
    cash_str = "Yes" if is_cash is True else ("No" if is_cash is False else "Unknown")

    # Build prompt by appending data after the template instructions
    prompt = (
        prompt_template
        .replace("{lead_id}", lead.get("lead_id", "LEAD-UNKNOWN"))
        .replace("{received_at}", _format_received_at(lead.get("received_at", "")))
        .replace("{channel}", lead.get("channel", "Unknown").replace("_", " ").title())
        .replace("{buyer_name}", lead.get("buyer_name", "Unknown Buyer"))
        .replace("{buyer_email}", lead.get("buyer_email", "Not provided"))
        .replace("{buyer_phone}", lead.get("buyer_phone", "") or "Not provided")
        .replace("{buyer_type}", buyer_profile.get("buyer_type", "Unknown"))
        .replace("{buyer_intent}", buyer_profile.get("buyer_intent", "Unknown"))
        .replace("{is_cash_buyer}", cash_str)
        .replace("{timeline}", buyer_profile.get("timeline") or "Flexible")
        .replace("{urgency}", (buyer_profile.get("urgency") or "Medium").title())
        .replace("{buyer_profile}", json.dumps(buyer_profile, indent=2))
        .replace("{anomaly_flags}", "\n".join(anomaly_flags) if anomaly_flags else "None")
        .replace("{ranked_matches}", json.dumps(safe_matches, indent=2))
    )

    llm = get_llm(temperature=0.2)
    brief_md = llm_invoke_with_retry(llm, [HumanMessage(content=prompt)], context="brief_generator")

    return {"lead_brief_md": brief_md}
