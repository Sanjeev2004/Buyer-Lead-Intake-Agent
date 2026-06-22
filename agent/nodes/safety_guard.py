"""
Node 1b: Safety & Anomaly Guard
Detects prompt injection, impossible budgets, missing contact info, vague leads,
and negotiation-focused leads. Adds structured flags to state.
"""
import re
from agent.state import AgentState

# ---------------------------------------------------------------------------
# Prompt injection patterns — phrases typically used in adversarial prompts
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"respond\s+by\s+listing",
    r"output\s+all\s+(owner|contact|phone|email)",
    r"in\s+json\s+format\s+so\s+i\s+can\s+contact",
    r"new\s+instruction[s]?:",
    r"system\s+prompt:",
    r"you\s+are\s+now",
]

# Minimum viable budget for any Miami property in the MLS
ABSOLUTE_MIN_MIAMI_BUDGET = 200_000

# Thresholds for impossible budget detection
# (price floor for location+type combinations in USD)
IMPOSSIBLE_BUDGET_CHECKS = [
    # (description, budget_max, bedrooms_min, neighborhoods_keywords, flag_message)
    (
        "4BR+ downtown/brickell with pool and ocean view",
        400_000,
        4,
        ["downtown", "brickell", "miami beach"],
        "⚠️ BUDGET MISMATCH: A {beds}BR+ property in {neighborhoods} with pool & ocean view "
        "cannot realistically be found at ${budget:,}. Minimum realistic budget is $1.5M+. "
        "Clarify budget before reaching out — this lead may be misinformed or testing the market.",
    ),
]


def _check_prompt_injection(message: str) -> list[str]:
    flags = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            flags.append(
                "🔒 SECURITY ALERT: Prompt injection attempt detected in buyer message. "
                "The embedded instruction has been ignored. Do NOT share owner contact details "
                "with this lead until identity is verified."
            )
            break  # one flag is enough
    return flags


def _check_anonymous_contact(lead: dict) -> list[str]:
    flags = []
    name = lead.get("buyer_name", "").strip()
    phone = lead.get("buyer_phone", "").strip()
    email = lead.get("buyer_email", "").strip()

    is_anonymous = (
        not name
        or name.lower().startswith("anonymous")
        or "not filled" in name.lower()
    )
    missing_phone = not phone
    generic_email = bool(re.match(r"^[a-z]{2,4}\d*@(gmail|yahoo|hotmail|outlook)", email, re.I)) if email else False

    if is_anonymous:
        flags.append(
            "⚠️ CONTACT: Anonymous lead — no real name provided. "
            "Cannot personalize outreach. Request name before sending property suggestions."
        )
    if missing_phone:
        flags.append(
            "⚠️ CONTACT: No phone number provided. Email-only outreach. "
            "Ask for a call-back number in first response."
        )
    return flags


def _check_impossible_budget(buyer_profile: dict) -> list[str]:
    flags = []
    budget_max = buyer_profile.get("budget_max")
    beds_min = buyer_profile.get("bedrooms_min") or 0
    neighborhoods = [n.lower() for n in (buyer_profile.get("neighborhoods") or [])]
    required_features = [f.lower() for f in (buyer_profile.get("required_features") or [])]

    if budget_max is None:
        return flags

    for desc, budget_threshold, beds_threshold, neighborhood_keywords, flag_template in IMPOSSIBLE_BUDGET_CHECKS:
        neighborhood_hit = any(kw in " ".join(neighborhoods) for kw in neighborhood_keywords)
        if (
            budget_max <= budget_threshold
            and beds_min >= beds_threshold
            and neighborhood_hit
        ):
            flags.append(
                flag_template.format(
                    beds=beds_min,
                    neighborhoods=", ".join(buyer_profile.get("neighborhoods", ["downtown Miami"])),
                    budget=budget_max,
                )
            )

    return flags


def _check_vague_lead(buyer_profile: dict) -> list[str]:
    flags = []
    has_budget = buyer_profile.get("budget_max") or buyer_profile.get("budget_min")
    has_beds = buyer_profile.get("bedrooms_min") or buyer_profile.get("bedrooms_max")
    has_neighborhoods = bool(buyer_profile.get("neighborhoods"))
    has_type = bool(buyer_profile.get("property_types") and buyer_profile["property_types"] != ["any"])

    detail_count = sum([bool(has_budget), bool(has_beds), bool(has_neighborhoods), bool(has_type)])

    if detail_count <= 1:
        flags.append(
            "⚠️ VAGUE LEAD: Insufficient detail to match properties. "
            "This lead needs a discovery call before property matching. "
            "See 'Clarifying Questions' section in the brief below."
        )
    return flags


def _check_negotiation_focus(message: str, buyer_profile: dict) -> list[str]:
    flags = []
    negotiation_patterns = [
        r"put\s+in\s+an?\s+offer",
        r"offer\s+at\s+\$",
        r"go\s+lower",
        r"sellers?[''s]?\s+motivation",
        r"asking\s+(price|is)\s+around",
        r"what\s+do\s+you\s+think\s+(about\s+)?(the\s+)?offer",
        r"should\s+i\s+offer",
    ]
    for pattern in negotiation_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            flags.append(
                "🤝 NEGOTIATION ADVISORY: This buyer is already focused on a specific listing "
                "and asking for offer strategy. Prepare a Comparative Market Analysis (CMA) "
                "for the target property before calling. Buyer appears ready to transact."
            )
            break
    return flags


def safety_guard_node(state: AgentState) -> AgentState:
    """
    Runs anomaly detection on the lead and buyer profile.
    Adds structured flag strings to state["anomaly_flags"].
    """
    lead = state["lead"]
    buyer_profile = state["buyer_profile"] or {}
    message = lead.get("message", "")

    flags = []
    flags.extend(_check_prompt_injection(message))
    flags.extend(_check_anonymous_contact(lead))
    flags.extend(_check_impossible_budget(buyer_profile))
    flags.extend(_check_vague_lead(buyer_profile))
    flags.extend(_check_negotiation_focus(message, buyer_profile))

    return {"anomaly_flags": flags}
