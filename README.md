# Buyer Lead Intake Agent

An AI-powered agent that processes incoming real estate buyer inquiries and produces structured **Lead Briefs** for realtors — built for the AgentMira engineering case study.

## What it does

1. Takes a buyer's free-text inquiry message
2. Extracts a structured buyer profile (intent, budget, bedrooms, neighborhoods, features, urgency)
3. Detects anomalies (prompt injection, impossible budgets, vague leads, negotiation focus)
4. Matches against a 299-listing Miami MLS dataset using rule-based pre-filtering + LLM re-ranking
5. Generates a rich Markdown Lead Brief the realtor can act on immediately

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent Framework | LangGraph |
| LLM | `llama-3.3-70b-versatile` via Groq API |
| Data Processing | pandas |
| Output Format | Markdown |

## Setup

### 1. Prerequisites

- Python 3.11 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Clone / download

```bash
cd buyer-lead-intake-agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Edit `.env` and replace `your_groq_api_key_here` with your actual Groq API key.

### 5. Run

```bash
python main.py
```

This will:
- Process all 12 buyer leads
- Write `output/brief_LEAD-2026-001.md` through `output/brief_LEAD-2026-012.md`
- Write `output/summary_index.md` — a one-page triage table

## Project Structure

```
buyer-lead-intake-agent/
├── main.py                          # Entry point
├── requirements.txt
├── .env.example
├── agent/
│   ├── graph.py                     # LangGraph state machine
│   ├── state.py                     # AgentState TypedDict
│   ├── nodes/
│   │   ├── parse_classify.py        # Node 1: LLM extracts buyer profile
│   │   ├── safety_guard.py          # Node 1b: anomaly + injection detection
│   │   ├── rule_filter.py           # Node 2: pandas MLS pre-filter
│   │   ├── llm_rerank.py            # Node 3: LLM re-ranks candidates
│   │   └── brief_generator.py       # Node 4: generates Markdown brief
│   └── prompts/
│       ├── parse_prompt.txt
│       ├── rerank_prompt.txt
│       └── brief_prompt.txt
├── data/
│   ├── miami_mls_listings.csv
│   └── sample_buyer_inquiries.json
└── output/                          # Generated Lead Briefs (auto-created)
    ├── summary_index.md
    └── brief_LEAD-2026-*.md
```

## Agent Pipeline

```
Buyer Inquiry
     │
     ▼
[Node 1] Parse & Classify ─── LLM extracts structured buyer profile
     │
     ▼
[Node 1b] Safety Guard ─────── Detects prompt injection, impossible budgets,
     │                         anonymous contacts, vague leads, negotiation asks
     ▼
[Node 2] Rule Filter ────────── pandas: budget/bedroom/neighborhood/type filter
     │                          → up to 20 candidates
     ▼
[Node 3] LLM Re-Rank ────────── LLM scores each candidate 0-10 → top 5 matches
     │
     ▼
[Node 4] Brief Generator ────── LLM writes the Markdown Lead Brief
     │
     ▼
output/brief_LEAD-2026-XXX.md
```

## Edge Cases Handled

| Lead | Issue | How Handled |
|---|---|---|
| LEAD-003 | Anonymous + impossible $250K budget for 4BR downtown | Both flags raised; realistic budget range shown |
| LEAD-004 | Vague — no detail beyond "investment property" | Classified as vague; brief shows clarifying questions |
| LEAD-005 | Buyer asking for offer negotiation strategy | Negotiation advisory flag; CMA recommendation |
| LEAD-006 | Prompt injection attack in message | Security flag raised; injected instruction scrubbed; owner PII protected |
| LEAD-008 | Extremely verbose/chatty message (Jennifer Walsh) | LLM extracts real requirements cleanly |
| LEAD-009 | Cash buyer | Surfaced prominently as 🟢 Cash Buyer |
| LEAD-010 | $8M luxury lead — limited MLS matches | Dataset limitation noted; recommends expanded search |
| LEAD-012 | Investor wanting 2-3 properties | Portfolio-style brief with multiple options |

## Output Format

Each Lead Brief contains:
- **Buyer Snapshot** — contact info, intent, cash status, urgency
- **What They're Looking For** — concise summary of needs
- **Realtor Alerts** — anomaly flags with actionable guidance
- **Top Property Matches** — scored properties with strengths, concerns, and negotiation notes
- **Suggested Next Action** — concrete next steps for the realtor

## Cost & Rate Limits (Groq Free Tier)

Processing all 12 leads uses approximately **80,000–100,000 tokens** total.

Groq free-tier limits for `llama-3.3-70b-versatile`:

| Limit | Value |
|---|---|
| Tokens per Minute (TPM) | 12,000 |
| Requests per Minute (RPM) | 30 |
| **Tokens per Day (TPD)** | **100,000** |

The agent handles **TPM rate limits** automatically with exponential backoff retry.  
If you hit the **daily quota (TPD)**, the agent will print a clear error and you must wait until midnight UTC to reset.

> **Tip**: Run `python main.py` fresh each day. With the 3-second inter-lead pause, all 12 leads typically complete in **5–8 minutes** on a fresh daily quota.

