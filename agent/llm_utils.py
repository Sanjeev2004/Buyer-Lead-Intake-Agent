"""
Shared LLM utility — supports multiple providers via a single config.

Provider selection (set in .env):
  LLM_PROVIDER=groq          → Groq API (default, free tier)
  LLM_PROVIDER=openrouter    → OpenRouter (access 100+ models, one key)
  LLM_PROVIDER=openai        → OpenAI directly

OpenRouter lets you switch models without rewriting any code — just change
LLM_MODEL in .env to any model slug from https://openrouter.ai/models

Examples:
  LLM_MODEL=meta-llama/llama-3.3-70b-instruct   (free on OpenRouter)
  LLM_MODEL=openai/gpt-4o-mini
  LLM_MODEL=anthropic/claude-3.5-sonnet
  LLM_MODEL=google/gemini-flash-1.5
  LLM_MODEL=mistralai/mistral-7b-instruct        (free on OpenRouter)
"""
import os
import sys
import time
import random
import re

from langchain_core.messages import HumanMessage

# ── Provider / Model defaults ──────────────────────────────────────────────────
PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower().strip()

_DEFAULTS = {
    "groq":        "llama-3.3-70b-versatile",
    "openrouter":  "meta-llama/llama-3.3-70b-instruct",
    "openai":      "gpt-4o-mini",
}
MODEL = os.environ.get("LLM_MODEL", _DEFAULTS.get(PROVIDER, "llama-3.3-70b-versatile"))

MAX_RETRIES = 6
BASE_DELAY  = 8   # seconds


def _build_groq_llm(temperature: float):
    """Groq-native client (langchain_groq)."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=MODEL,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


def _build_openrouter_llm(temperature: float):
    """
    OpenRouter via langchain_openai — OpenAI-compatible endpoint.
    Supports every model on https://openrouter.ai/models via one API key.

    To pin to a specific backend, set OPENROUTER_PROVIDER_SUFFIX in .env:
      cerebras  → Cerebras wafer-scale (~2000 tok/s, ultra-fast)
      groq      → Groq backend
      free      → free tier (any provider)
    e.g. OPENROUTER_PROVIDER_SUFFIX=cerebras
    """
    from langchain_openai import ChatOpenAI

    suffix = os.environ.get("OPENROUTER_PROVIDER_SUFFIX", "").strip()
    model_id = f"{MODEL}:{suffix}" if suffix else MODEL

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/Sanjeev2004/Buyer-Lead-Intake-Agent",
            "X-Title": "Buyer Lead Intake Agent",
        },
    )


def _build_openai_llm(temperature: float):
    """Direct OpenAI client."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=MODEL,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )


_BUILDERS = {
    "groq":       _build_groq_llm,
    "openrouter": _build_openrouter_llm,
    "openai":     _build_openai_llm,
}


def get_llm(temperature: float = 0):
    """
    Returns a configured LLM client for the active provider.

    Provider is chosen by LLM_PROVIDER env var (default: groq).
    Model is chosen by LLM_MODEL env var (default: provider-specific).
    """
    builder = _BUILDERS.get(PROVIDER)
    if builder is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{PROVIDER}'. "
            f"Valid options: {list(_BUILDERS.keys())}"
        )
    return builder(temperature)


def llm_invoke_with_retry(llm, messages: list, context: str = "") -> str:
    """
    Invoke the LLM with exponential backoff retry on rate limit errors (429).
    Works for Groq, OpenRouter, and OpenAI clients.
    Returns the response content string.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            err_str = str(e)

            # ── Groq-specific: daily quota exhaustion — never retry ──────────
            is_daily_quota = (
                "tokens per day" in err_str.lower()
                or "TPD" in err_str
            )
            if is_daily_quota:
                raise RuntimeError(
                    "Groq free-tier daily token quota (100K TPD) exhausted. "
                    "Wait until midnight UTC for the quota to reset, or:\n"
                    "  • Upgrade Groq: https://console.groq.com/settings/billing\n"
                    "  • Switch to OpenRouter: set LLM_PROVIDER=openrouter in .env"
                ) from e

            # ── Rate limit (429) — retry with exponential backoff ────────────
            is_rate_limit = (
                "429" in err_str
                or "rate_limit" in err_str.lower()
                or "rate limit" in err_str.lower()
                or "tokens per minute" in err_str.lower()
                or "requests per minute" in err_str.lower()
                or "too many requests" in err_str.lower()
            )
            is_last_attempt = attempt == MAX_RETRIES

            if is_rate_limit and not is_last_attempt:
                wait = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(1, 3)

                # Honour provider's suggested retry-after if present
                retry_match = re.search(r"try again in (\d+\.?\d*)s", err_str)
                if retry_match:
                    wait = max(float(retry_match.group(1)) + 2.0, wait)

                wait = min(wait, 65.0)   # cap at one TPM window

                msg = (
                    f"\n    [Rate limit] {context} ({PROVIDER}/{MODEL}) "
                    f"— attempt {attempt}/{MAX_RETRIES}, waiting {wait:.0f}s..."
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
                time.sleep(wait)
            else:
                raise
