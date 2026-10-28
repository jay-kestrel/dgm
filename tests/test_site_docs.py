"""
Unit tests for documentation site link translation and HTML documentation integrity.
"""

import re
import sys
from pathlib import Path

# Add scripts directory to path to import build_docs_site functions
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_docs_site import DOCS_DIR, SITE_DOCS_DIR, rewrite_markdown_links  # noqa: E402


def test_rewrite_markdown_links_overview():
    """Verify link rewriting in docs/index.md for the 'Documentation Contents' section."""
    index_src = DOCS_DIR / "index.md"
    html_input = (
        '<ul>\n'
        '<li><a href="walkthrough.md">Walkthrough</a></li>\n'
        '<li><a href="architecture.md">Architecture</a></li>\n'
        '<li><a href="configuration.md">Configuration</a></li>\n'
        '<li><a href="api_reference.md">API Reference</a></li>\n'
        '<li><a href="tutorials/01_quickstart.md">Tutorial 1</a></li>\n'
        '<li><a href="tutorials/02_ensemble_clouds.md">Tutorial 2</a></li>\n'
        '<li><a href="../AGENTS.md">AGENTS</a></li>\n'
        '<li><a href="https://example.com/other.md">External</a></li>\n'
        '</ul>'
    )

    rewritten = rewrite_markdown_links(html_input, index_src)

    assert 'href="walkthrough.html"' in rewritten
    assert 'href="architecture.html"' in rewritten
    assert 'href="configuration.html"' in rewritten
    assert 'href="api_reference.html"' in rewritten
    assert 'href="tutorial_01_quickstart.html"' in rewritten
    assert 'href="tutorial_02_ensemble_clouds.html"' in rewritten
    assert 'href="https://github.com/jay-kestrel/dgm/blob/main/AGENTS.md"' in rewritten
    assert 'href="https://example.com/other.md"' in rewritten


def test_rewrite_markdown_links_with_anchors_and_nested_paths():
    """Verify anchor preservation and nested path translation."""
    config_src = DOCS_DIR / "configuration.md"
    html_input = '<p><a href="api_reference.md#dgmconfig">DGMConfig</a></p>'
    rewritten = rewrite_markdown_links(html_input, config_src)
    assert 'href="api_reference.html#dgmconfig"' in rewritten

    tutorial_src = DOCS_DIR / "tutorials" / "01_quickstart.md"
    tut_input = (
        '<p><a href="../walkthrough.md">Walkthrough</a></p>'
        '<p><a href="02_ensemble_clouds.md">Ensembles</a></p>'
    )
    tut_rewritten = rewrite_markdown_links(tut_input, tutorial_src)
    assert 'href="walkthrough.html"' in tut_rewritten
    assert 'href="tutorial_02_ensemble_clouds.html"' in tut_rewritten


def test_site_docs_html_links_exist():
    """Verify that all compiled site/docs/*.html files contain valid, existing relative links."""
    assert SITE_DOCS_DIR.exists()

    href_pat = re.compile(r'href=["\']([^"\']+)["\']')

    for html_file in SITE_DOCS_DIR.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        for match in href_pat.finditer(content):
            href = match.group(1)

            # Skip external links, fragments, javascript, and query parameters
            if href.startswith(("http://", "https://", "mailto:", "javascript:", "#")):
                continue
            if href.startswith("../index.html"):
                assert (SITE_DOCS_DIR.parent / "index.html").exists()
                continue

            target_part = href.split("#")[0].split("?")[0]
            if not target_part:
                continue

            # Target file must end with .html, not .md
            assert not target_part.endswith(".md"), (
                f"Broken .md link found in {html_file.name}: {href}"
            )

            target_file = (html_file.parent / target_part).resolve()
            assert target_file.exists(), (
                f"Dead link in {html_file.name}: {href} -> {target_file} not found"
            )
