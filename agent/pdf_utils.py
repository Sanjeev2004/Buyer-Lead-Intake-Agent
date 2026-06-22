# -*- coding: utf-8 -*-
"""
pdf_utils.py — Converts Markdown content to styled PDF files.

Uses the pipeline: Markdown → HTML → PDF (via xhtml2pdf).
Produces premium-styled PDFs with AgentMira branding.
"""
import io
from pathlib import Path

import markdown as md_lib
from xhtml2pdf import pisa


# ──────────────────────────────────────────────────────────────────
# CSS — Premium real-estate brand styling
# ──────────────────────────────────────────────────────────────────
PDF_CSS = """
@page {
    size: A4;
    margin: 18mm 18mm 18mm 18mm;
    @frame footer {
        -pdf-frame-content: footer_content;
        bottom: 8mm;
        margin-left: 18mm;
        margin-right: 18mm;
        height: 10mm;
    }
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    color: #1a1a2e;
    line-height: 1.55;
    background: #ffffff;
}

/* ── Header / Title ── */
h1 {
    font-size: 18pt;
    font-weight: bold;
    color: #0f3460;
    border-bottom: 3px solid #e94560;
    padding-bottom: 4pt;
    margin-bottom: 6pt;
}

h2 {
    font-size: 13pt;
    font-weight: bold;
    color: #16213e;
    background-color: #eef2ff;
    padding: 4pt 8pt;
    border-left: 4pt solid #e94560;
    margin-top: 14pt;
    margin-bottom: 6pt;
}

h3 {
    font-size: 11pt;
    font-weight: bold;
    color: #0f3460;
    margin-top: 10pt;
    margin-bottom: 4pt;
}

/* ── Paragraph & body text ── */
p {
    margin: 4pt 0 6pt 0;
}

/* ── Bold / strong ── */
strong {
    color: #0f3460;
}

/* ── Horizontal rule ── */
hr {
    border: none;
    border-top: 1.5pt solid #e94560;
    margin: 10pt 0;
}

/* ── Tables (Buyer Snapshot & Summary Index) ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 6pt 0 10pt 0;
    font-size: 9.5pt;
}

th {
    background-color: #0f3460;
    color: #ffffff;
    font-weight: bold;
    padding: 5pt 8pt;
    text-align: left;
    border: 1pt solid #0f3460;
}

td {
    padding: 4pt 8pt;
    border: 0.5pt solid #c8d0e0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background-color: #f4f6fb;
}

/* ── Unordered / Ordered lists ── */
ul, ol {
    margin: 4pt 0 6pt 16pt;
    padding: 0;
}

li {
    margin-bottom: 2pt;
}

/* ── Code / pre (for MLS# etc.) ── */
code {
    font-family: Courier, monospace;
    font-size: 9pt;
    background-color: #f0f0f0;
    padding: 1pt 3pt;
}

/* ── Links ── */
a {
    color: #e94560;
    text-decoration: none;
}

/* ── Footer content div ── */
#footer_content {
    font-size: 7.5pt;
    color: #888888;
    text-align: center;
    border-top: 0.5pt solid #cccccc;
    padding-top: 2pt;
}
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <style>{css}</style>
</head>
<body>
  <div id="footer_content">AgentMira · Buyer Lead Intake Agent · Confidential</div>
  {body}
</body>
</html>
"""


def markdown_to_pdf(markdown_text: str, output_path: Path) -> None:
    """
    Converts a Markdown string to a styled PDF file at output_path.

    Args:
        markdown_text: The full Markdown content (e.g., a Lead Brief).
        output_path:   Destination .pdf file path (parent dir must exist).

    Raises:
        RuntimeError: If xhtml2pdf reports a conversion error.
    """
    # Step 1: Markdown → HTML
    # 'tables' extension handles the Buyer Snapshot table
    html_body = md_lib.markdown(
        markdown_text,
        extensions=["tables", "nl2br", "sane_lists"],
    )

    # Step 2: Wrap in full HTML document with CSS
    full_html = HTML_TEMPLATE.format(css=PDF_CSS, body=html_body)

    # Step 3: HTML → PDF via xhtml2pdf
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as pdf_file:
        result = pisa.CreatePDF(
            src=io.StringIO(full_html),
            dest=pdf_file,
            encoding="utf-8",
        )

    if result.err:
        raise RuntimeError(
            f"xhtml2pdf reported {result.err} error(s) converting to {output_path.name}"
        )
