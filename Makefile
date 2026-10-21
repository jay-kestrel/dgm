# DeltaGraphModel (DGM) Makefile
# Standardized developer commands for testing, documentation, and client-side RAG portal

.PHONY: help install test lint site-build site-docs site-index site-serve clean

# Default target: display help
help:
	@echo "DeltaGraphModel (DGM) Build & Development Commands:"
	@echo "  make install     - Sync virtual environment and install dev dependencies via uv"
	@echo "  make test        - Run complete pytest suite across all modules"
	@echo "  make lint        - Run ruff linter checks on codebase"
	@echo "  make site-build  - Full documentation portal build (HTML docs + RAG index + embeddings)"
	@echo "  make site-docs   - Compile only markdown docs into Kestrel-styled HTML pages"
	@echo "  make site-index  - Regenerate client-side BM25 search index and vector embeddings"
	@echo "  make site-serve  - Launch local HTTP server at http://localhost:8000 to preview portal"
	@echo "  make clean       - Remove cached files, test artifacts, and build outputs"

install:
	uv sync --extra dev

test:
	uv run --extra dev pytest tests -v

lint:
	uv run ruff check src tests

site-build:
	uv run --with fastembed,markdown python scripts/build_docs_site.py

site-docs:
	uv run --with markdown python scripts/build_docs_site.py --docs-only

site-index:
	uv run --with fastembed python scripts/build_docs_site.py --index-only

site-serve:
	@echo "Starting local documentation & AI assistant portal at http://localhost:8000 (Ctrl+C to stop)..."
	python3 -m http.server 8000 -d site

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
