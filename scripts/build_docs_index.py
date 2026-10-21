#!/usr/bin/env python3
"""
build_docs_index.py
Builds a static JSON index of the DGM codebase, documentation, and configuration
for fast client-side in-browser semantic search and WebLLM Q&A.
"""

import ast
import json
import re
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO_URL = "https://github.com/jay-kestrel/dgm/blob/main"


def extract_python_chunks(py_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    rel_path = py_path.relative_to(REPO_ROOT).as_posix()
    code_text = py_path.read_text(encoding="utf-8")
    lines = code_text.splitlines()

    try:
        tree = ast.parse(code_text)
    except Exception as e:
        print(f"Warning: could not parse {rel_path}: {e}")
        return chunks

    # Module-level docstring
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        chunks.append({
            "id": f"{rel_path}:module",
            "title": f"{py_path.stem} (Module Overview)",
            "category": "Code: Module",
            "type": "module",
            "path": rel_path,
            "lines": "1-20",
            "summary": mod_doc.strip().split("\n\n")[0],
            "content": f"Module: {rel_path}\n\n{mod_doc.strip()}",
            "github_url": f"{GITHUB_REPO_URL}/{rel_path}"
        })

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            start = node.lineno
            end = getattr(node, "end_lineno", start + 20)
            snippet = "\n".join(lines[start - 1:end])
            first_sentence = doc.strip().split("\n")[0] if doc else f"Function {node.name}"
            # Extract arguments
            args = [a.arg for a in node.args.args]
            sig = f"{node.name}({', '.join(args)})"

            chunks.append({
                "id": f"{rel_path}:{node.name}",
                "title": sig,
                "category": f"Code: {py_path.parent.name}",
                "type": "function",
                "path": rel_path,
                "lines": f"{start}-{end}",
                "summary": first_sentence,
                "content": f"Function `{sig}` in `{rel_path}` (lines {start}-{end}):\nDocstring:\n{doc}\n\nCode Snippet:\n```python\n{snippet[:1200]}\n```",
                "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{start}-L{end}"
            })

        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            start = node.lineno
            end = getattr(node, "end_lineno", start + 30)
            first_sentence = doc.strip().split("\n")[0] if doc else f"Class {node.name}"
            methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]

            chunks.append({
                "id": f"{rel_path}:{node.name}",
                "title": f"class {node.name}",
                "category": f"Code: {py_path.parent.name}",
                "type": "class",
                "path": rel_path,
                "lines": f"{start}-{end}",
                "summary": first_sentence,
                "content": f"Class `{node.name}` in `{rel_path}` (methods: {', '.join(methods)}):\nDocstring:\n{doc}",
                "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{start}-L{end}"
            })

            # Also index class methods
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_doc = ast.get_docstring(m) or ""
                    m_start = m.lineno
                    m_end = getattr(m, "end_lineno", m_start + 15)
                    m_args = [a.arg for a in m.args.args]
                    m_sig = f"{node.name}.{m.name}({', '.join(m_args)})"
                    chunks.append({
                        "id": f"{rel_path}:{node.name}.{m.name}",
                        "title": m_sig,
                        "category": f"Code: {node.name}",
                        "type": "method",
                        "path": rel_path,
                        "lines": f"{m_start}-{m_end}",
                        "summary": m_doc.strip().split("\n")[0] if m_doc else f"Method {m_sig}",
                        "content": f"Method `{m_sig}`:\n{m_doc}",
                        "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{m_start}-L{m_end}"
                    })

    return chunks


def extract_markdown_chunks(md_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    rel_path = md_path.relative_to(REPO_ROOT).as_posix()
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Split by markdown headers (# , ## , ### )
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")
    current_title = md_path.stem.replace("_", " ").title()
    current_lines: List[str] = []
    current_start = 1

    def save_chunk(title: str, content_lines: List[str], start_line: int, end_line: int):
        body = "\n".join(content_lines).strip()
        if len(body) < 30:
            return
        first_para = body.split("\n\n")[0].replace("\n", " ")
        summary = first_para[:160] + "..." if len(first_para) > 160 else first_para
        chunks.append({
            "id": f"{rel_path}:{re.sub(r'[^a-zA-Z0-9_]+', '-', title).lower()}",
            "title": title,
            "category": "Documentation",
            "type": "doc",
            "path": rel_path,
            "lines": f"{start_line}-{end_line}",
            "summary": summary,
            "content": f"Documentation: {rel_path} > {title}\n\n{body[:2000]}",
            "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{start_line}-L{end_line}"
        })

    for idx, line in enumerate(lines, start=1):
        match = header_pattern.match(line)
        if match:
            if current_lines:
                save_chunk(current_title, current_lines, current_start, idx - 1)
            current_title = match.group(2).strip()
            current_lines = [line]
            current_start = idx
        else:
            current_lines.append(line)

    if current_lines:
        save_chunk(current_title, current_lines, current_start, len(lines))

    return chunks


def extract_yaml_chunks(yaml_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    rel_path = yaml_path.relative_to(REPO_ROOT).as_posix()
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    current_section = "General"
    section_lines: Dict[str, List[str]] = {}

    section_header_pattern = re.compile(r"^#\s*---\s*Step\s*\d+:\s*(.+?)\s*---")
    top_key_pattern = re.compile(r"^([a-z_]+):")

    for line in lines:
        s_match = section_header_pattern.match(line)
        if s_match:
            current_section = s_match.group(1).strip()
            if current_section not in section_lines:
                section_lines[current_section] = []
            continue
        k_match = top_key_pattern.match(line)
        if k_match and not line.startswith(" "):
            key = k_match.group(1)
            current_section = key.capitalize()
            if current_section not in section_lines:
                section_lines[current_section] = []
        if current_section in section_lines:
            section_lines[current_section].append(line)

    for section, sec_lines in section_lines.items():
        body = "\n".join(sec_lines).strip()
        if not body:
            continue
        chunks.append({
            "id": f"{rel_path}:{section.lower().replace(' ', '_')}",
            "title": f"Config: {section}",
            "category": "Configuration",
            "type": "config",
            "path": rel_path,
            "lines": "1-end",
            "summary": f"YAML configuration parameters for {section}",
            "content": f"Configuration schema for {section} in `{rel_path}`:\n```yaml\n{body}\n```",
            "github_url": f"{GITHUB_REPO_URL}/{rel_path}"
        })

    return chunks


def build_index():
    chunks: List[Dict[str, Any]] = []

    # 1. Python source files in src/dgm
    py_files = sorted((REPO_ROOT / "src" / "dgm").glob("**/*.py"))
    for py_file in py_files:
        if py_file.name == "__pycache__":
            continue
        chunks.extend(extract_python_chunks(py_file))

    # 2. Markdown documentation in docs/ and README
    md_files = sorted((REPO_ROOT / "docs").glob("**/*.md"))
    md_files.append(REPO_ROOT / "README.md")
    for md_file in md_files:
        if md_file.exists():
            chunks.extend(extract_markdown_chunks(md_file))

    # 3. YAML config
    cfg_file = REPO_ROOT / "config" / "default_config.yaml"
    if cfg_file.exists():
        chunks.extend(extract_yaml_chunks(cfg_file))

    # 4. Examples
    example_files = sorted((REPO_ROOT / "examples").glob("*.py"))
    for ex_file in example_files:
        chunks.extend(extract_python_chunks(ex_file))

    out_dir = REPO_ROOT / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "search_index.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Indexed {len(chunks)} chunks across code, docs, and config.")
    print(f"Saved index to: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_index()
