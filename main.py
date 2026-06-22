# -*- coding: utf-8 -*-
"""
main.py — Entry point for the Buyer Lead Intake Agent.

Usage:
    python main.py

Reads all leads from data/sample_buyer_inquiries.json,
runs each through the LangGraph pipeline,
and writes styled PDF Lead Briefs to output/brief_LEAD-2026-XXX.pdf

Also writes output/summary_index.pdf — a triage table for all 12 leads.
"""
import json
import os
import sys
import time
from pathlib import Path

from agent.pdf_utils import markdown_to_pdf

import io
from dotenv import load_dotenv

# Force UTF-8 output on Windows so emoji don't crash CP1252 consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load environment variables from .env file
load_dotenv()

# Validate Groq API key
if not os.environ.get("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY is not set.")
    print("   Copy .env.example to .env and add your Groq API key.")
    print("   Get a free key at: https://console.groq.com")
    sys.exit(1)

from agent.graph import build_graph

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
LEADS_FILE = DATA_DIR / "sample_buyer_inquiries.json"


def load_leads() -> list[dict]:
    with open(LEADS_FILE, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(lead: dict, graph) -> str:
    """Runs a single lead through the full LangGraph pipeline."""
    initial_state = {
        "lead": lead,
        "buyer_profile": None,
        "anomaly_flags": [],
        "mls_candidates": [],
        "ranked_matches": [],
        "lead_brief_md": "",
    }
    result = graph.invoke(initial_state)
    return result["lead_brief_md"]


def write_brief(lead_id: str, brief_md: str) -> Path:
    """Converts Markdown Lead Brief to a styled PDF in the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / f"brief_{lead_id}.pdf"
    markdown_to_pdf(brief_md, filename)
    return filename


def write_summary_index(results: list[dict]) -> Path:
    """Writes a styled PDF summary triage table of all leads."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build Markdown — links point to .pdf brief files
    lines = [
        "# Lead Briefs — Summary Index",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')} | Total Leads: {len(results)}",
        "",
        "| Lead ID | Buyer | Channel | Flags | Status |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        lead = r["lead"]
        flags = r["flags"]
        status = r["status"]
        flag_str = " ".join(flags[:2]) if flags else "Clean"
        lines.append(
            f"| {lead['lead_id']} "
            f"| {lead['buyer_name']} "
            f"| {lead['channel'].replace('_', ' ').title()} "
            f"| {flag_str[:60]} "
            f"| {status} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Flag Legend",
        "- Security alert (prompt injection)",
        "- Anomaly (impossible budget, anonymous, vague)",
        "- Negotiation advisory",
        "- Clean lead",
    ]

    summary_md = "\n".join(lines)
    summary_path = OUTPUT_DIR / "summary_index.pdf"
    markdown_to_pdf(summary_md, summary_path)
    return summary_path


def main():
    print("=" * 60)
    print("  Buyer Lead Intake Agent — AgentMira Case Study")
    print("  Model: llama-3.3-70b-versatile via Groq")
    print("=" * 60)
    print()

    leads = load_leads()
    print(f"📋 Loaded {len(leads)} leads from {LEADS_FILE.name}")
    print()

    graph = build_graph()
    results = []

    for i, lead in enumerate(leads, 1):
        lead_id = lead.get("lead_id", f"LEAD-{i:03d}")
        buyer_name = lead.get("buyer_name", "Unknown")
        print(f"[{i:02d}/{len(leads)}] Processing {lead_id} — {buyer_name}... ", end="", flush=True)

        start = time.time()
        try:
            brief_md = run_pipeline(lead, graph)
            elapsed = time.time() - start

            # Extract flags from the brief for the summary index
            flags = []
            if "🔒 SECURITY" in brief_md:
                flags.append("🔒")
            if "⚠️" in brief_md:
                flags.append("⚠️")
            if "🤝 NEGOTIATION" in brief_md:
                flags.append("🤝")

            output_path = write_brief(lead_id, brief_md)
            print(f"✅ Done ({elapsed:.1f}s) → {output_path.name} (PDF)")

            results.append({
                "lead": lead,
                "flags": flags,
                "status": "✅ Generated",
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ Failed ({elapsed:.1f}s): {e}")
            results.append({
                "lead": lead,
                "flags": ["❌"],
                "status": f"❌ Error: {str(e)[:40]}",
            })

        # Brief inter-lead pause to respect Groq free-tier TPM limits (12k tokens/min)
        # Skip pause after the last lead
        if i < len(leads):
            time.sleep(3)

    print()
    summary_path = write_summary_index(results)
    print(f"📄 Summary index written → {summary_path.name} (PDF)")
    print()

    success_count = sum(1 for r in results if "✅" in r["status"])
    print(f"{'=' * 60}")
    print(f"  ✅ {success_count}/{len(leads)} Lead Briefs generated successfully")
    print(f"  📁 Output folder: {OUTPUT_DIR.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
