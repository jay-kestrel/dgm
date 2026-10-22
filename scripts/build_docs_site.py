#!/usr/bin/env python3
"""
build_docs_site.py
Compiles DGM markdown documentation in docs/ into static HTML pages matching the
Kestrel Intelligence dark tactical HUD design system, and updates the search index
and vector embeddings for the client-side AI Assistant.
"""

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "site"
SITE_DOCS_DIR = SITE_DIR / "docs"

EXCLUDED_DOC_FILES: set[str] = set()

# Navigation manifest for documentation sidebar & header
DOCS_MANIFEST = [
    {
        "slug": "index",
        "title": "Overview",
        "category": "Getting Started",
        "src": DOCS_DIR / "index.md",
        "dest": "index.html",
        "prompt": "Explain the high-level capabilities, philosophy, and installation of DeltaGraphModel.",
    },
    {
        "slug": "architecture",
        "title": "Architecture & Math",
        "category": "Core Theory",
        "src": DOCS_DIR / "architecture.md",
        "dest": "architecture.html",
        "prompt": "Explain the 7 core pipeline stages, Murray's law hydraulics, and loop engines in DGM.",
    },
    {
        "slug": "configuration",
        "title": "YAML Configuration",
        "category": "Core Theory",
        "src": DOCS_DIR / "configuration.md",
        "dest": "configuration.html",
        "prompt": "How do DGM parameters, stochastic distributions, and validation constraints work in default_config.yaml?",
    },
    {
        "slug": "api_reference",
        "title": "Python API Reference",
        "category": "Reference",
        "src": DOCS_DIR / "api_reference.md",
        "dest": "api_reference.html",
        "prompt": "What are the main functions and classes in DGM public API (dgm.generate, MetricsBattery, etc.)?",
    },
    {
        "slug": "parameter_justifications",
        "title": "Parameter Justifications",
        "category": "Reference",
        "src": DOCS_DIR / "parameter_justifications.md",
        "dest": "parameter_justifications.html",
        "prompt": "What are the scientific justifications, empirical foundations, and literature citations for DGM parameters?",
    },
    {
        "slug": "walkthrough",
        "title": "End-to-End Walkthrough",
        "category": "Tutorials",
        "src": DOCS_DIR / "walkthrough.md",
        "dest": "walkthrough.html",
        "prompt": "Walk me through generating a delta, routing flow, and computing metrics step by step in Python.",
    },
    {
        "slug": "tutorial_01_quickstart",
        "title": "Tutorial 01: Quickstart",
        "category": "Tutorials",
        "src": DOCS_DIR / "tutorials" / "01_quickstart.md",
        "dest": "tutorial_01_quickstart.html",
        "prompt": "How do I run the DGM quickstart example and export to GraphML or GeoJSON?",
    },
    {
        "slug": "tutorial_02_ensemble_clouds",
        "title": "Tutorial 02: Ensembles",
        "category": "Tutorials",
        "src": DOCS_DIR / "tutorials" / "02_ensemble_clouds.md",
        "dest": "tutorial_02_ensemble_clouds.html",
        "prompt": "How does DGM sample large stochastic ensembles and compute geomorphic metric distributions?",
    },
]


def preprocess_markdown(raw_md: str) -> tuple[str, dict[str, str]]:
    """
    Protects LaTeX math expressions and code blocks from being mangled by Python-Markdown
    (e.g., prevents underscores inside math from turning into italic tags).
    """
    math_store: dict[str, str] = {}
    counter = 0

    code_store: dict[str, str] = {}
    code_counter = 0

    def code_repl(m: re.Match[str]) -> str:
        nonlocal code_counter
        k = f"DGM_CODE_BLOCK_{code_counter}_X"
        code_counter += 1
        code_store[k] = m.group(0)
        return k

    # 1. Protect fenced code blocks (```...```) and inline code (`...`)
    raw_md = re.sub(r"```[\s\S]*?```", code_repl, raw_md)
    raw_md = re.sub(r"`[^`\n]+`", code_repl, raw_md)

    # 2. Protect display math: $$ ... $$ (can span multiple lines)
    def disp_repl(m: re.Match[str]) -> str:
        nonlocal counter
        k = f"DGM_MATH_DISP_{counter}_X"
        counter += 1
        math_store[k] = m.group(0)
        return f"\n\n<div class=\"katex-display-block\">{k}</div>\n\n"

    raw_md = re.sub(r"\$\$([\s\S]+?)\$\$", disp_repl, raw_md)

    # 3. Protect inline math: $ ... $ (single-line, non-empty, not preceded/followed by $)
    def inl_repl(m: re.Match[str]) -> str:
        nonlocal counter
        k = f"DGM_MATH_INL_{counter}_X"
        counter += 1
        math_store[k] = m.group(0)
        return f"<span class=\"katex-inline-block\">{k}</span>"

    raw_md = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", inl_repl, raw_md)

    # 4. Restore code blocks
    for k, v in code_store.items():
        raw_md = raw_md.replace(k, v)

    return raw_md, math_store


def postprocess_html(html_text: str, math_store: dict[str, str]) -> str:
    """
    Restores original math expressions, wraps tables in responsive containers,
    and converts GitHub-style alert callouts into styled Kestrel HUD cards.
    """
    # 1. Restore math expressions
    for k, orig_math in math_store.items():
        html_text = html_text.replace(k, orig_math)

    # 2. Transform GitHub-style blockquote callouts ([!NOTE], [!TIP], [!IMPORTANT], [!WARNING], [!CAUTION])
    callout_cfg: dict[str, dict[str, str]] = {
        "NOTE": {"color": "#38bdf8", "border": "#38bdf8", "bg": "rgba(56, 189, 248, 0.07)", "label": "NOTE"},
        "TIP": {"color": "#c4f94a", "border": "#c4f94a", "bg": "rgba(196, 249, 74, 0.07)", "label": "TIP"},
        "IMPORTANT": {"color": "#ff5a35", "border": "#ff5a35", "bg": "rgba(255, 90, 53, 0.07)", "label": "IMPORTANT"},
        "WARNING": {"color": "#f59e0b", "border": "#f59e0b", "bg": "rgba(245, 158, 11, 0.07)", "label": "WARNING"},
        "CAUTION": {"color": "#f43f5e", "border": "#f43f5e", "bg": "rgba(244, 63, 94, 0.07)", "label": "CAUTION"},
    }

    def callout_repl(m: re.Match[str]) -> str:
        kind = m.group(1).upper()
        content = m.group(2).strip()
        c = callout_cfg.get(kind, callout_cfg["NOTE"])
        return f"""<div class="my-6 border-l-2 p-4 text-xs font-mono" style="border-color: {c["border"]}; background: {c["bg"]};">
  <div class="flex items-center gap-2 mb-2 font-bold tracking-wider uppercase text-[10px]" style="color: {c["color"]};">
    <span>[ {c["label"]} ]</span>
  </div>
  <div class="text-white/85 font-sans leading-relaxed text-sm">
    {content}
  </div>
</div>"""

    callout_pat = re.compile(
        r"<blockquote>\s*<p>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\s*<br\s*/?>)?\s*(.*?)</p>\s*</blockquote>",
        re.DOTALL | re.IGNORECASE
    )
    html_text = callout_pat.sub(callout_repl, html_text)

    # 3. Wrap tables in responsive container for clean horizontal scrolling
    html_text = re.sub(
        r"(<table>[\s\S]*?</table>)",
        r'<div class="table-container overflow-x-auto my-6 border border-white/10">\1</div>',
        html_text
    )

    return html_text


def render_html_page(doc_info: dict[str, Any], body_html: str) -> str:
    current_dest = doc_info["dest"]
    title = doc_info["title"]
    category = doc_info["category"]
    prompt = doc_info["prompt"]

    # Build Sidebar links grouped by category
    sidebar_items_html = []
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in DOCS_MANIFEST:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    for cat_name, items in categories.items():
        sidebar_items_html.append(
            f'<div class="text-[10px] font-mono uppercase tracking-widest text-white/40 px-3 pt-3 pb-1 border-t border-white/[0.08] first:border-t-0">'
            f"{cat_name}"
            f"</div>"
        )
        for item in items:
            is_active = item["dest"] == current_dest
            active_cls = (
                "bg-white/[0.08] text-[#ff5a35] border-l-2 border-l-[#ff5a35] font-semibold"
                if is_active
                else "text-white/70 hover:text-white hover:bg-white/[0.03] border-l-2 border-l-transparent"
            )
            sidebar_items_html.append(
                f'<a href="{item["dest"]}" class="block px-3 py-1.5 text-xs transition {active_cls}">{html.escape(item["title"])}</a>'
            )

    sidebar_html = "\n".join(sidebar_items_html)

    return f"""<!DOCTYPE html>
<html lang="en" class="h-full bg-[#0a0b0a] text-white">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)} | DeltaGraphModel Documentation</title>

  <!-- Kestrel Typography: Aldrich (Display), Azeret Mono (Code/HUD), Plus Jakarta Sans (Body) -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Aldrich&family=Azeret+Mono:ital,wght@0,300..700;1,300..700&family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&display=swap" rel="stylesheet">

  <!-- Tailwind CSS via CDN -->
  <script src="https://cdn.tailwindcss.com"></script>

  <!-- KaTeX for High-Precision Mathematical Typography -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>

  <!-- Mermaid.js for Interactive Diagrams -->
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{
      startOnLoad: false,
      theme: 'dark',
      securityLevel: 'loose',
      themeVariables: {{
        darkMode: true,
        background: '#060706',
        primaryColor: '#ff5a35',
        primaryTextColor: '#ffffff',
        primaryBorderColor: 'rgba(255, 90, 53, 0.6)',
        lineColor: '#c4f94a',
        secondaryColor: '#0c0e0c',
        tertiaryColor: '#121512',
        fontFamily: '"Plus Jakarta Sans", sans-serif'
      }}
    }});

    document.addEventListener('DOMContentLoaded', async () => {{
      // 1. Render KaTeX Math
      if (typeof renderMathInElement === 'function') {{
        renderMathInElement(document.body, {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
          ],
          throwOnError: false,
          trust: true
        }});
      }}

      // 2. Render Mermaid Diagrams
      const codeBlocks = document.querySelectorAll('pre code.language-mermaid');
      for (const block of codeBlocks) {{
        const pre = block.parentElement;
        const div = document.createElement('div');
        div.className = 'mermaid my-6 flex justify-center bg-[#060706] p-4 border border-white/10 overflow-x-auto';
        div.textContent = block.textContent;
        pre.replaceWith(div);
      }}
      if (codeBlocks.length > 0) {{
        await mermaid.run();
      }}
    }});
  </script>

  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            kestrel: {{
              bg: '#0a0b0a',
              panel: '#0c0e0c',
              terminal: '#060706',
              border: 'rgba(255, 255, 255, 0.1)',
              orange: '#ff5a35',
              lime: '#c4f94a',
              teal: '#cfddd0',
            }}
          }},
          fontFamily: {{
            heading: ['Aldrich', 'sans-serif'],
            mono: ['"Azeret Mono"', 'monospace'],
            sans: ['"Plus Jakarta Sans"', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
          }}
        }}
      }}
    }}
  </script>

  <style>
    *, ::before, ::after {{
      border-radius: 0px !important;
    }}
    body {{
      background-color: #0a0b0a;
      color: #ffffff;
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }}
    .teal-ambient {{
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      background:
        radial-gradient(circle at 50% -12%, rgba(207, 221, 208, 0.08) 0%, transparent 65%),
        radial-gradient(circle at 90% 10%, rgba(255, 90, 53, 0.04) 0%, transparent 45%);
      pointer-events: none;
      z-index: 0;
    }}
    /* Custom square scrollbars */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: #070807; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255, 255, 255, 0.15); }}
    ::-webkit-scrollbar-thumb:hover {{ background: #ff5a35; }}

    /* Kestrel Buttons */
    .kestrel-btn {{
      font-family: 'Azeret Mono', monospace;
      font-size: 0.6875rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 500;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: rgba(255, 255, 255, 0.05);
      color: #fff;
      padding: 0.45rem 0.85rem;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
    }}
    .kestrel-btn:hover {{
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.4);
    }}
    .kestrel-btn-orange {{
      background: #ff5a35;
      border-color: #ff5a35;
      color: #fff;
      font-weight: 600;
      box-shadow: 0 0 14px rgba(255, 90, 53, 0.3);
    }}
    .kestrel-btn-orange:hover {{
      background: #ff7a5a;
      border-color: #ff7a5a;
      box-shadow: 0 0 20px rgba(255, 90, 53, 0.45);
    }}

    /* Markdown prose formatting */
    .prose-kestrel h1 {{
      font-family: 'Aldrich', sans-serif;
      font-size: 1.75rem;
      line-height: 1.25;
      color: #ffffff;
      margin-top: 1.5rem;
      margin-bottom: 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 0.5rem;
    }}
    .prose-kestrel h2 {{
      font-family: 'Aldrich', sans-serif;
      font-size: 1.25rem;
      line-height: 1.35;
      color: #ffffff;
      margin-top: 2rem;
      margin-bottom: 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 0.35rem;
    }}
    .prose-kestrel h3 {{
      font-family: 'Aldrich', sans-serif;
      font-size: 1rem;
      color: #ffffff;
      margin-top: 1.5rem;
      margin-bottom: 0.5rem;
    }}
    .prose-kestrel h4 {{
      font-family: 'Azeret Mono', monospace;
      font-size: 0.8rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #c4f94a;
      margin-top: 1.25rem;
      margin-bottom: 0.5rem;
    }}
    .prose-kestrel p {{
      margin-bottom: 1rem;
      line-height: 1.7;
      color: rgba(255, 255, 255, 0.75);
      font-size: 0.875rem;
    }}
    .prose-kestrel ul, .prose-kestrel ol {{
      margin-left: 1.5rem;
      margin-bottom: 1.25rem;
      color: rgba(255, 255, 255, 0.75);
      font-size: 0.875rem;
      line-height: 1.7;
    }}
    .prose-kestrel ul {{ list-style-type: square; }}
    .prose-kestrel ol {{ list-style-type: decimal; }}
    .prose-kestrel li {{ margin-bottom: 0.35rem; }}
    .prose-kestrel pre {{
      background: #050605;
      border: 1px solid rgba(255, 255, 255, 0.12);
      padding: 1rem;
      overflow-x: auto;
      margin: 1.25rem 0;
      font-family: 'Azeret Mono', monospace;
      font-size: 0.75rem;
      line-height: 1.55;
      color: #e2e8f0;
    }}
    .prose-kestrel code {{
      background: rgba(255, 255, 255, 0.08);
      color: #c4f94a;
      padding: 0.15rem 0.35rem;
      font-family: 'Azeret Mono', monospace;
      font-size: 0.75rem;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .prose-kestrel pre code {{
      background: transparent;
      color: #e2e8f0;
      padding: 0;
      border: none;
    }}

    /* Table Improvements & Container */
    .prose-kestrel .table-container {{
      overflow-x: auto;
      margin: 1.5rem 0;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: #070907;
    }}
    .prose-kestrel table {{
      width: 100%;
      border-collapse: collapse;
      margin: 0;
      font-size: 0.8rem;
    }}
    .prose-kestrel th {{
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-bottom: 2px solid rgba(255, 255, 255, 0.2);
      padding: 0.65rem 0.85rem;
      text-align: left;
      font-family: 'Azeret Mono', monospace;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #ffffff;
      white-space: nowrap;
    }}
    .prose-kestrel td {{
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 0.65rem 0.85rem;
      color: rgba(255, 255, 255, 0.8);
      vertical-align: top;
    }}
    .prose-kestrel tr:nth-child(even) td {{
      background: rgba(255, 255, 255, 0.015);
    }}
    .prose-kestrel tr:hover td {{
      background: rgba(255, 255, 255, 0.035);
    }}

    /* KaTeX Mathematical Formula Styling */
    .katex-display-block {{
      margin: 1.25rem 0;
      padding: 0.75rem 1rem;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.08);
      overflow-x: auto;
    }}
    .katex-display {{
      margin: 0.5rem 0 !important;
      overflow-x: auto;
      overflow-y: hidden;
    }}
    .katex {{
      color: #ffffff;
      font-size: 1.05em;
    }}

    /* Mermaid Diagram Styling */
    .mermaid {{
      background: #060706;
      border: 1px solid rgba(255, 255, 255, 0.12);
      padding: 1.5rem;
      margin: 1.5rem 0;
      display: flex;
      justify-content: center;
      overflow-x: auto;
    }}
    .mermaid svg {{
      max-width: 100%;
      height: auto;
    }}

    .prose-kestrel blockquote {{
      border-left: 2px solid #ff5a35;
      background: rgba(255, 90, 53, 0.05);
      padding: 0.75rem 1rem;
      margin: 1.25rem 0;
      font-size: 0.85rem;
      color: rgba(255, 255, 255, 0.85);
    }}
    .prose-kestrel a {{
      color: #ff5a35;
      text-decoration: underline;
    }}
    .prose-kestrel hr {{
      border: none;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      margin: 2rem 0;
    }}
  </style>
</head>
<body class="h-full flex flex-col font-sans antialiased text-white bg-[#0a0b0a] relative selection:bg-[#ff5a35]/30 selection:text-white">

  <!-- Ambient Light Layer -->
  <div class="teal-ambient" aria-hidden="true"></div>

  <!-- Header Navigation -->
  <header class="border-b border-white/[0.1] bg-[#0a0b0a] sticky top-0 z-50 shrink-0">
    <div class="w-full px-4 h-14 flex items-center justify-between">

      <!-- Brandmark & Section -->
      <div class="flex items-center space-x-3">
        <a href="../index.html" class="w-8 h-8 bg-white/[0.04] border border-white/[0.2] text-[#ff5a35] flex items-center justify-center font-heading font-bold text-base hover:border-[#ff5a35] transition" title="Go to AI Assistant Front Door">
          Δ
        </a>
        <div class="flex items-center space-x-2">
          <a href="../index.html" class="font-heading text-sm sm:text-base tracking-wider text-white hover:text-[#ff5a35] transition">
            DeltaGraphModel
          </a>
          <span class="text-white/30 font-mono text-xs">/</span>
          <span class="text-xs font-mono text-[#ff5a35] uppercase tracking-wider">
            {html.escape(title)}
          </span>
        </div>
      </div>

      <!-- Header Actions -->
      <div class="flex items-center space-x-3">
        <a href="../index.html?q={html.escape(prompt)}" class="kestrel-btn kestrel-btn-orange text-[10px] py-1 px-3" title="Ask DGM Assistant about this document">
          <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 14h-2v-2h2v2zm0-4h-2V7h2v5z"/></svg>
          <span>Ask AI About This</span>
        </a>
        <a href="../index.html" class="kestrel-btn text-[10px] py-1 px-3">
          <span>AI Terminal &rarr;</span>
        </a>
      </div>

    </div>
  </header>

  <!-- Main Body: Sidebar + Reading Canvas -->
  <div class="flex-1 w-full flex overflow-hidden relative z-10">

    <!-- Left Navigation Sidebar -->
    <aside class="w-64 shrink-0 border-r border-white/[0.1] bg-[#0c0e0c] h-full overflow-y-auto hidden md:block">
      <div class="p-3 border-b border-white/[0.1] bg-[#080908] flex items-center justify-between">
        <span class="text-[10px] font-mono uppercase tracking-widest text-white/50">Documentation</span>
        <span class="text-[9px] font-mono text-[#c4f94a]">[ VERIFIED ]</span>
      </div>
      <nav class="py-2">
        {sidebar_html}
      </nav>
      <div class="p-4 mt-6 border-t border-white/[0.08] bg-white/[0.01]">
        <div class="text-[10px] font-mono text-white/40 uppercase mb-2">Need Instant Answers?</div>
        <a href="../index.html" class="block w-full text-center kestrel-btn text-[10px] py-2">
          Open AI Terminal
        </a>
      </div>
    </aside>

    <!-- Reading Canvas -->
    <main class="flex-1 h-full overflow-y-auto px-6 py-8 md:px-12 lg:px-16 bg-[#0a0b0a]">
      <div class="max-w-4xl mx-auto">

        <!-- Breadcrumbs -->
        <div class="flex items-center space-x-2 text-[10px] font-mono uppercase tracking-wider text-white/40 mb-4 pb-2 border-b border-white/[0.06]">
          <a href="index.html" class="hover:text-white transition">Docs</a>
          <span>&gt;</span>
          <span>{html.escape(category)}</span>
          <span>&gt;</span>
          <span class="text-[#ff5a35]">{html.escape(title)}</span>
        </div>

        <!-- Markdown Body -->
        <article class="prose-kestrel">
          {body_html}
        </article>

        <!-- Page Footer / Next Step Banner -->
        <div class="mt-12 pt-6 border-t border-white/[0.1] flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs font-mono text-white/50">
          <div>
            Grounding Source for <strong class="text-white">DGM Hybrid RAG</strong>
          </div>
          <div class="flex items-center space-x-3">
            <a href="../index.html?q={html.escape(prompt)}" class="kestrel-btn kestrel-btn-orange text-[10px] py-1 px-3">
              Ask AI About This Topic
            </a>
            <a href="https://github.com/jay-kestrel/dgm" target="_blank" class="kestrel-btn text-[10px] py-1 px-3">
              View Source on GitHub
            </a>
          </div>
        </div>

      </div>
    </main>

  </div>

</body>
</html>
"""


def build_docs_html():
    try:
        import markdown
    except ImportError:
        print("Error: 'markdown' package not installed. Run with `uv run --with markdown python scripts/build_docs_site.py`")
        sys.exit(1)

    SITE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling markdown documentation to Kestrel HTML pages...")
    rendered_count = 0

    for doc in DOCS_MANIFEST:
        src_path = doc["src"]
        dest_filename = doc["dest"]
        dest_path = SITE_DOCS_DIR / dest_filename

        if not src_path.exists():
            print(f"Warning: source file not found: {src_path}")
            continue

        if src_path.name in EXCLUDED_DOC_FILES:
            print(f"Skipping excluded document: {src_path.name}")
            continue

        raw_md = src_path.read_text(encoding="utf-8")

        # 1. Preprocess: protect math expressions and code blocks
        preprocessed_md, math_store = preprocess_markdown(raw_md)

        # 2. Convert markdown to HTML
        body_html = markdown.markdown(preprocessed_md, extensions=["fenced_code", "tables", "toc", "sane_lists"])

        # 3. Postprocess: restore math, style callouts and wrap tables
        body_html = postprocess_html(body_html, math_store)

        full_html = render_html_page(doc, body_html)
        dest_path.write_text(full_html, encoding="utf-8")
        print(f"  ✓ {src_path.name} -> site/docs/{dest_filename} ({dest_path.stat().st_size / 1024:.1f} KB)")
        rendered_count += 1

    print(f"Successfully compiled {rendered_count} documentation pages in {SITE_DOCS_DIR}.")


def main():
    parser = argparse.ArgumentParser(description="Build complete DGM documentation site & AI search index.")
    parser.add_argument("--docs-only", action="store_true", help="Compile only the HTML documentation pages.")
    parser.add_argument("--index-only", action="store_true", help="Compile only the search index & vector embeddings.")
    args = parser.parse_args()

    # 1. Compile HTML documentation pages
    if not args.index_only:
        build_docs_html()

    # 2. Build Search Index & Float16 Vector Embeddings
    if not args.docs_only:
        print("\nRebuilding Search Index and Vector Embeddings...")
        from build_docs_index import build_index

        build_index()

    print("\n[COMPLETE] DGM documentation portal & AI assistant build finished.")


if __name__ == "__main__":
    main()
