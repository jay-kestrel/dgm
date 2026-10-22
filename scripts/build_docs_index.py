#!/usr/bin/env python3
"""
build_docs_index.py
Builds a static JSON index and precomputed float16 vector embeddings of the DGM
codebase, documentation, empirical delta benchmarks, and configuration for fast
client-side in-browser hybrid search (BM25 + Semantic Vectors) and Transformers.js Q&A.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO_URL = "https://github.com/jay-kestrel/dgm/blob/main"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Documents excluded from search indexing (if any)
EXCLUDED_DOC_FILES: set[str] = set()


def extract_python_chunks(py_path: Path) -> list[dict[str, Any]]:
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
        chunks.append(
            {
                "id": f"{rel_path}:module",
                "title": f"{py_path.stem} (Module Overview)",
                "category": "Code: Module",
                "type": "module",
                "path": rel_path,
                "lines": "1-20",
                "summary": mod_doc.strip().split("\n\n")[0],
                "content": f"Module: {rel_path}\n\n{mod_doc.strip()}",
                "github_url": f"{GITHUB_REPO_URL}/{rel_path}",
            }
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            start = node.lineno
            end = getattr(node, "end_lineno", start + 20)
            snippet = "\n".join(lines[start - 1 : end])
            first_sentence = doc.strip().split("\n")[0] if doc else f"Function {node.name}"
            args = [a.arg for a in node.args.args]
            sig = f"{node.name}({', '.join(args)})"

            chunks.append(
                {
                    "id": f"{rel_path}:{node.name}",
                    "title": sig,
                    "category": f"Code: {py_path.parent.name}",
                    "type": "function",
                    "path": rel_path,
                    "lines": f"{start}-{end}",
                    "summary": first_sentence,
                    "content": f"Function `{sig}` in `{rel_path}` (lines {start}-{end}):\nDocstring:\n{doc}\n\nCode Snippet:\n```python\n{snippet[:1500]}\n```",
                    "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{start}-L{end}",
                }
            )

        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            start = node.lineno
            end = getattr(node, "end_lineno", start + 30)
            first_sentence = doc.strip().split("\n")[0] if doc else f"Class {node.name}"
            methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]

            # Include class header + fields (e.g., for dataclasses like DGMConfig)
            class_header_end = min(end, start + 45)
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_header_end = min(class_header_end, m.lineno - 1)
                    break
            class_snippet = "\n".join(lines[start - 1 : max(start, class_header_end)])

            chunks.append(
                {
                    "id": f"{rel_path}:{node.name}",
                    "title": f"class {node.name}",
                    "category": f"Code: {py_path.parent.name}",
                    "type": "class",
                    "path": rel_path,
                    "lines": f"{start}-{end}",
                    "summary": first_sentence,
                    "content": (
                        f"Class `{node.name}` in `{rel_path}` (methods: {', '.join(methods)}):\n"
                        f"Docstring:\n{doc}\n\nClass Definition:\n```python\n{class_snippet[:1200]}\n```"
                    ),
                    "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{start}-L{end}",
                }
            )

            # Also index class methods with full code snippets
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_doc = ast.get_docstring(m) or ""
                    m_start = m.lineno
                    m_end = getattr(m, "end_lineno", m_start + 15)
                    m_snippet = "\n".join(lines[m_start - 1 : m_end])
                    m_args = [a.arg for a in m.args.args]
                    m_sig = f"{node.name}.{m.name}({', '.join(m_args)})"
                    chunks.append(
                        {
                            "id": f"{rel_path}:{node.name}.{m.name}",
                            "title": m_sig,
                            "category": f"Code: {node.name}",
                            "type": "method",
                            "path": rel_path,
                            "lines": f"{m_start}-{m_end}",
                            "summary": m_doc.strip().split("\n")[0] if m_doc else f"Method {m_sig}",
                            "content": (
                                f"Method `{m_sig}` in `{rel_path}` (lines {m_start}-{m_end}):\n"
                                f"Docstring:\n{m_doc}\n\nCode Snippet:\n```python\n{m_snippet[:1500]}\n```"
                            ),
                            "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{m_start}-L{m_end}",
                        }
                    )

    return chunks


def extract_markdown_chunks(md_path: Path) -> list[dict[str, Any]]:
    chunks = []
    rel_path = md_path.relative_to(REPO_ROOT).as_posix()
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Split by markdown headers (# , ## , ### )
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")
    current_title = md_path.stem.replace("_", " ").title()
    current_lines: list[str] = []
    current_start = 1

    def save_chunk(title: str, content_lines: list[str], start_line: int, end_line: int):
        body = "\n".join(content_lines).strip()
        if len(body) < 30:
            return
        first_para = body.split("\n\n")[0].replace("\n", " ")
        summary = first_para[:160] + "..." if len(first_para) > 160 else first_para

        # Map markdown file to rendered HTML documentation page
        html_dest = "index.html"
        if md_path.name == "architecture.md":
            html_dest = "architecture.html"
        elif md_path.name == "configuration.md":
            html_dest = "configuration.html"
        elif md_path.name == "api_reference.md":
            html_dest = "api_reference.html"
        elif md_path.name == "walkthrough.md":
            html_dest = "walkthrough.html"
        elif "01_quickstart" in md_path.name:
            html_dest = "tutorial_01_quickstart.html"
        elif "02_ensemble" in md_path.name:
            html_dest = "tutorial_02_ensemble_clouds.html"

        chunks.append(
            {
                "id": f"{rel_path}:{re.sub(r'[^a-zA-Z0-9_]+', '-', title).lower()}-{start_line}",
                "title": title,
                "category": "Documentation",
                "type": "doc",
                "path": rel_path,
                "lines": f"{start_line}-{end_line}",
                "summary": summary,
                "content": f"Documentation: {rel_path} > {title}\n\n{body[:2000]}",
                "doc_url": f"docs/{html_dest}",
                "github_url": f"{GITHUB_REPO_URL}/{rel_path}#L{start_line}-L{end_line}",
            }
        )

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


def extract_yaml_chunks(yaml_path: Path) -> list[dict[str, Any]]:
    chunks = []
    rel_path = yaml_path.relative_to(REPO_ROOT).as_posix()
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    current_section = "General"
    section_lines: dict[str, list[str]] = {}

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
        chunks.append(
            {
                "id": f"{rel_path}:{section.lower().replace(' ', '_')}",
                "title": f"Config: {section}",
                "category": "Configuration",
                "type": "config",
                "path": rel_path,
                "lines": "1-end",
                "summary": f"YAML configuration parameters for {section}",
                "content": f"Configuration schema for {section} in `{rel_path}`:\n```yaml\n{body}\n```",
                "github_url": f"{GITHUB_REPO_URL}/{rel_path}",
            }
        )

    return chunks


def extract_empirical_delta_chunks() -> list[dict[str, Any]]:
    """Extracts the 28 canonical empirical deltas from DIDNT dataset into searchable chunks."""
    canonical_file = REPO_ROOT / "src" / "dgm" / "data" / "canonical_empirical_deltas.json"
    if not canonical_file.exists():
        return []

    rel_path = canonical_file.relative_to(REPO_ROOT).as_posix()
    try:
        deltas = json.loads(canonical_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not parse {rel_path}: {e}")
        return []

    chunks = []
    # Summary table chunk of all 28 deltas
    overview_lines = ["Empirical Delta Benchmark Dataset (28 Global Deltas from DIDNT / Hariharan et al., 2021):"]
    for d in deltas:
        overview_lines.append(
            f"- {d['name']}: N={d['nodes']} nodes, E={d['edges']} edges, loops={d['loops']}, "
            f"forcing={d['forcing_type']}, R_rt={d['estimated_R_rt']}, nER*={d['nER_star']:.3f}, "
            f"alpha={d['alpha']:.3f}, outlets={d['num_outlets']}"
        )

    chunks.append(
        {
            "id": f"{rel_path}:all_deltas_summary",
            "title": "28 Empirical Deltas (DIDNT Benchmark Summary)",
            "category": "Dataset: DIDNT",
            "type": "dataset",
            "path": rel_path,
            "lines": "1-end",
            "summary": "Complete table of 28 real-world river deltas (nodes, loops, R_rt, forcing regime, nER*, outlets).",
            "content": "\n".join(overview_lines),
            "github_url": f"{GITHUB_REPO_URL}/{rel_path}",
        }
    )

    for d in deltas:
        name = d["name"]
        chunks.append(
            {
                "id": f"{rel_path}:delta_{name.lower().replace(' ', '_')}",
                "title": f"Delta: {name} ({d['forcing_type']})",
                "category": "Dataset: DIDNT",
                "type": "dataset",
                "path": rel_path,
                "lines": "1-end",
                "summary": (
                    f"{name} Delta ({d['forcing_type']}, R_rt={d['estimated_R_rt']}): "
                    f"N={d['nodes']} nodes, E={d['edges']} edges, {d['loops']} loops, {d['num_outlets']} outlets."
                ),
                "content": (
                    f"Empirical Delta Metrics for {name} Delta (DIDNT Dataset):\n"
                    f"- Forcing Regime: {d['forcing_type']} (estimated R_rt = {d['estimated_R_rt']})\n"
                    f"- Topology: N = {d['nodes']} nodes, E = {d['edges']} links, Betti-1 = {d['loops']} loops, "
                    f"Outlets = {d['num_outlets']}\n"
                    f"- Loopiness: Meshedness alpha = {d['alpha']:.4f}, Normalized Betti-1 = {d['betti_1_norm']:.4f}, "
                    f"Vc/Vb Confluence-to-Bifurcation ratio = {d['vc_vb_ratio']:.4f}\n"
                    f"- Spectral & Energetic: nER* = {d['nER_star']:.4f}, Algebraic Connectivity = {d['algebraic_connectivity']:.5f}, "
                    f"Resistance Distance = {d['resistance_distance']:.4f}, Link Sharing Index = {d['link_sharing_index']:.4f}\n"
                    f"- Flux & Geometry: eBI = {d['eBI']:.3f}, Outlet Flux Entropy = {d['outlet_flux_entropy']:.4f}, "
                    f"Width CV = {d['width_cv']:.4f}, Radial Sinuosity = {d['radial_sinuosity']:.4f}"
                ),
                "github_url": f"{GITHUB_REPO_URL}/{rel_path}",
            }
        )

    return chunks


def compute_and_save_embeddings(chunks: list[dict[str, Any]], out_dir: Path) -> None:
    """Computes normalized 384-d embeddings using all-MiniLM-L6-v2 and saves as compact float16 binary."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print(
            "Warning: 'fastembed' not installed. Run with `uv run --with fastembed python scripts/build_docs_index.py` "
            "to regenerate embeddings.bin."
        )
        return

    print(f"Computing 384-d semantic embeddings for {len(chunks)} chunks using {EMBEDDING_MODEL_NAME}...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    texts = [f"{c['title']} ({c['category']})\n{c['summary']}\n{c['content'][:1000]}" for c in chunks]
    vecs = list(model.embed(texts))
    mat = np.array(vecs, dtype=np.float32)

    # Normalize to unit length so cosine similarity == dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    mat = (mat / norms).astype(np.float16)

    emb_bin_path = out_dir / "embeddings.bin"
    mat.tofile(emb_bin_path)

    meta_path = out_dir / "embeddings_meta.json"
    meta = {
        "model": "Xenova/all-MiniLM-L6-v2",
        "count": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "dtype": "float16",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved binary embeddings to: {emb_bin_path} ({emb_bin_path.stat().st_size / 1024:.1f} KB, shape={mat.shape}, dtype=float16)")


def build_index():
    chunks: list[dict[str, Any]] = []

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
        if md_file.exists() and md_file.name not in EXCLUDED_DOC_FILES:
            chunks.extend(extract_markdown_chunks(md_file))

    # 3. YAML config
    cfg_file = REPO_ROOT / "config" / "default_config.yaml"
    if cfg_file.exists():
        chunks.extend(extract_yaml_chunks(cfg_file))

    # 4. Examples
    example_files = sorted((REPO_ROOT / "examples").glob("*.py"))
    for ex_file in example_files:
        chunks.extend(extract_python_chunks(ex_file))

    # 5. Canonical Empirical Deltas (DIDNT dataset)
    chunks.extend(extract_empirical_delta_chunks())

    out_dir = REPO_ROOT / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "search_index.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Indexed {len(chunks)} chunks across code, docs, config, and empirical datasets.")
    print(f"Saved index to: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")

    compute_and_save_embeddings(chunks, out_dir)


if __name__ == "__main__":
    build_index()
