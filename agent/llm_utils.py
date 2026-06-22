"""
Shared LLM utility: creates a Groq LLM client with retry logic for rate limits.
"""
import os
import sys
import time
import random
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 6
BASE_DELAY = 8  # seconds — Groq free tier TPM resets every minute


def get_llm(temperature: float = 0) -> ChatGroq:
    """Returns a configured Groq LLM client."""
    return ChatGroq(
        model=MODEL,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


def llm_invoke_with_retry(llm: ChatGroq, messages: list, context: str = "") -> str:
    """
    Invoke the LLM with exponential backoff retry on rate limit errors (429).
    Returns the response content string.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            err_str = str(e)

            # Daily quota exhaustion — retrying won't help, raise immediately
            is_daily_quota = "tokens per day" in err_str.lower() or "TPD" in err_str
            if is_daily_quota:
                raise RuntimeError(
                    "Groq free-tier daily token quota (100K TPD) exhausted. "
                    "Wait until midnight UTC for the quota to reset, or upgrade at "
                    "https://console.groq.com/settings/billing"
                ) from e

            is_rate_limit = (
                "429" in err_str
                or "rate_limit" in err_str.lower()
                or "tokens per minute" in err_str.lower()
                or "requests per minute" in err_str.lower()
            )
            is_last_attempt = attempt == MAX_RETRIES

            if is_rate_limit and not is_last_attempt:
                # Parse Groq's suggested retry time from error message
                wait = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(1, 3)
                retry_match = re.search(r"try again in (\d+\.?\d*)s", err_str)
                if retry_match:
                    wait = max(float(retry_match.group(1)) + 2.0, wait)

                # Cap at 65 seconds per attempt (one TPM window)
                wait = min(wait, 65.0)

                msg = f"\n    [Rate limit] {context} — attempt {attempt}/{MAX_RETRIES}, waiting {wait:.0f}s..."
                sys.stdout.write(msg)
                sys.stdout.flush()
                time.sleep(wait)
            else:
                raise

