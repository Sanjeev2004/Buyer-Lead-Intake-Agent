"""
Node 2: Rule-Based Pre-Filter
Uses pandas to filter the MLS CSV down to a manageable shortlist (≤20 listings)
before handing off to the LLM re-ranker.
"""
import re
import pandas as pd
from pathlib import Path
from agent.state import AgentState

MLS_PATH = Path(__file__).parent.parent.parent / "data" / "miami_mls_listings.csv"

# Neighbourhood alias map: buyer-friendly names → substrings to match in CSV
NEIGHBORHOOD_ALIASES: dict[str, list[str]] = {
    "brickell": ["brickell"],
    "downtown miami": ["downtown", "brickell"],
    "downtown": ["downtown", "brickell"],
    "coral gables": ["coral gables"],
    "coconut grove": ["coconut grove"],
    "aventura": ["aventura"],
    "miami beach": ["miami beach"],
    "north miami": ["north miami"],
    "south beach": ["south beach", "miami beach"],
    "wynwood": ["wynwood", "midtown", "edgewater"],
    "edgewater": ["edgewater", "wynwood"],
    "midtown": ["midtown", "wynwood"],
    "key biscayne": ["key biscayne"],
    "bal harbour": ["bal harbour", "surfside", "sunny isles"],
    "pinecrest": ["pinecrest", "coral gables"],
    "hallandale": ["hallandale"],
    "doral": ["doral"],
    "miami gardens": ["miami gardens"],
    "hialeah": ["hialeah"],
    "south miami": ["south miami", "coral gables", "pinecrest"],
}


def _load_mls() -> pd.DataFrame:
    df = pd.read_csv(MLS_PATH, dtype=str)
    # Normalise numeric columns
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    df["days_on_market"] = pd.to_numeric(df["days_on_market"], errors="coerce")
    df["neighborhood_lower"] = df["neighborhood"].str.lower().fillna("")
    df["features_lower"] = df["features"].str.lower().fillna("")
    df["description_lower"] = df["description"].str.lower().fillna("")
    df["property_type_lower"] = df["property_type"].str.lower().fillna("")
    return df


def _neighborhood_filter(df: pd.DataFrame, neighborhoods: list[str]) -> pd.Series:
    """Returns a boolean mask for rows matching any of the buyer's preferred neighborhoods."""
    if not neighborhoods:
        return pd.Series([True] * len(df), index=df.index)

    mask = pd.Series([False] * len(df), index=df.index)
    for nbhd in neighborhoods:
        nbhd_lower = nbhd.lower().strip()
        # Direct substring match
        mask |= df["neighborhood_lower"].str.contains(nbhd_lower, na=False)
        # Alias expansion
        for alias_key, alias_values in NEIGHBORHOOD_ALIASES.items():
            if alias_key in nbhd_lower or nbhd_lower in alias_key:
                for av in alias_values:
                    mask |= df["neighborhood_lower"].str.contains(av, na=False)
    return mask


def _property_type_filter(df: pd.DataFrame, property_types: list[str]) -> pd.Series:
    """Returns a boolean mask for rows matching any of the buyer's property types."""
    if not property_types or property_types == ["any"]:
        return pd.Series([True] * len(df), index=df.index)

    mask = pd.Series([False] * len(df), index=df.index)
    type_map = {
        "condo": ["condo"],
        "single_family": ["single family"],
        "single family": ["single family"],
        "townhouse": ["townhouse", "town home", "townhome"],
        "multi_family": ["multi family", "multifamily", "duplex", "triplex"],
        "villa": ["villa"],
    }
    for pt in property_types:
        pt_lower = pt.lower().strip()
        for key, synonyms in type_map.items():
            if key in pt_lower or pt_lower in key:
                for syn in synonyms:
                    mask |= df["property_type_lower"].str.contains(syn, na=False)
        # Also do a direct match
        mask |= df["property_type_lower"].str.contains(pt_lower, na=False)
    return mask


def rule_filter_node(state: AgentState) -> AgentState:
    """
    Filters MLS listings using rule-based criteria from the buyer profile.
    Returns up to 20 best candidates for LLM re-ranking.
    """
    buyer_profile = state.get("buyer_profile") or {}
    anomaly_flags = state.get("anomaly_flags", [])

    # If lead is vague, don't try to match — return empty list
    if any("VAGUE LEAD" in f for f in anomaly_flags):
        return {"mls_candidates": []}

    df = _load_mls()

    # Only active listings
    mask_active = df["listing_status"].str.lower().str.contains("active", na=False)
    df = df[mask_active].copy()

    # ── Budget filter (10% flex above stated max) ──────────────────────────
    budget_max = buyer_profile.get("budget_max")
    budget_min = buyer_profile.get("budget_min")

    if budget_max:
        df = df[df["price"] <= budget_max * 1.10]
    if budget_min:
        df = df[df["price"] >= budget_min * 0.90]

    # ── Bedroom filter ─────────────────────────────────────────────────────
    beds_min = buyer_profile.get("bedrooms_min")
    beds_max = buyer_profile.get("bedrooms_max")

    if beds_min:
        df = df[df["bedrooms"] >= beds_min]
    if beds_max:
        df = df[df["bedrooms"] <= beds_max + 1]  # +1 flex

    # ── Neighborhood filter ────────────────────────────────────────────────
    neighborhoods = buyer_profile.get("neighborhoods") or []
    if neighborhoods:
        nbhd_mask = _neighborhood_filter(df, neighborhoods)
        neighborhood_df = df[nbhd_mask].copy()
        if len(neighborhood_df) == 0:
            # Neighbourhood not found — relax and take all
            neighborhood_df = df.copy()
    else:
        neighborhood_df = df.copy()

    # ── Property type filter ───────────────────────────────────────────────
    property_types = buyer_profile.get("property_types") or ["any"]
    type_mask = _property_type_filter(neighborhood_df, property_types)
    type_df = neighborhood_df[type_mask].copy()

    if len(type_df) == 0:
        # Type not found — fall back to neighbourhood-filtered set
        type_df = neighborhood_df.copy()

    # ── Required features soft filter ─────────────────────────────────────
    # Features are scored, not hard-filtered, to avoid over-filtering
    required_features = [f.lower() for f in (buyer_profile.get("required_features") or [])]
    type_df = type_df.copy()  # ensure it's never a view before we add columns
    if required_features:
        def feature_score(row) -> int:
            combined = str(row.get("features_lower", "")) + " " + str(row.get("description_lower", ""))
            return int(sum(1 for feat in required_features if feat.split()[0] in combined))

        scores = type_df.apply(feature_score, axis=1)
        type_df = type_df.copy()
        type_df["_feature_score"] = scores.values  # use .values to avoid index alignment issues
        # Sort by feature score descending, then by days on market ascending (fresher listings first)
        type_df = type_df.sort_values(["_feature_score", "days_on_market"], ascending=[False, True])
    else:
        type_df = type_df.sort_values("days_on_market", ascending=True)

    # Cap at 20 candidates
    candidates = type_df.head(20)

    # Drop internal-only PII columns before passing to LLM
    safe_cols = [c for c in candidates.columns if c not in ("owner_name", "owner_phone", "neighborhood_lower", "features_lower", "description_lower", "property_type_lower", "_feature_score")]
    candidates_safe = candidates[safe_cols].to_dict(orient="records")

    return {"mls_candidates": candidates_safe}
