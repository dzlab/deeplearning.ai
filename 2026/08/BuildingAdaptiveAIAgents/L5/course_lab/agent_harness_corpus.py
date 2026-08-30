"""Extract a code-search corpus from the REAL agent-harness source tree.

Module 4's embedding arm fine-tunes a bi-encoder on (intent-query -> gold chunk)
pairs. Until now its only corpus was the synthetic ``scm_codebase.json``. This
module produces a SECOND, real corpus by running the course's graphify parser
(``code_graph_parse``) over the actual ``agent-harness`` repo and emitting chunks
in the exact ``scm_codebase`` shape, so the embedding arm can train/eval on real
symbols with real docstrings.

Per-symbol facts are REAL (parsed from source):
  * ``signature`` — the real ``def name(args)`` line
  * ``docstring`` — the real first-line docstring (or a derived summary)
  * ``body``      — an AST-derived summary: the internal calls the function makes
  * ``imports`` / ``co_edits`` — real AST/git edges, FILTERED so every reference
    points at another chunk in this corpus (the loader enforces this)

Only the *queries* are synthetic (generated separately, live-Grok + cached) —
they must be, or the base encoder wins by lexical match and there is no headroom.

Committed fixture: ``course_lab/data/agent_harness_codebase.json`` (same
``{"chunks": [...]}`` envelope as ``scm_codebase.json``).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from course_lab import paths
from course_lab.code_graph_parse import (
    _DEFAULT_TREE,
    _call_name,
    _iter_py_files,
    _module_name,
    _rel,
    _resolve_module,
    _top_level_symbols,
)

_PKG_ROOT = "agent_harness"


def _signature(node: ast.AST) -> str:
    args = [a.arg for a in getattr(node, "args", ast.arguments()).args] \
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
    name = getattr(node, "name", "?")
    if isinstance(node, ast.ClassDef):
        return f"class {name}:"
    kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{kw} {name}({', '.join(args)}):"


def _docstring(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        doc = ast.get_docstring(node)
        if doc:
            return doc.strip().split("\n")[0][:200]
    return ""


def _body_summary(node: ast.AST, max_calls: int = 6) -> str:
    """AST-derived body: the internal calls the symbol makes, in order."""
    calls: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _call_name(sub.func)
            if name and name not in calls:
                calls.append(name)
        if len(calls) >= max_calls:
            break
    if calls:
        return "calls " + ", ".join(calls)
    return "no internal calls"


def build_corpus(tree: Path | None = None, *, max_chunks: int = 240) -> dict:
    """Parse the real tree into scm_codebase-shaped chunks. Deterministic.

    ``max_chunks`` caps the corpus to roughly the SCM fixture's size (219) so the
    two corpora are comparable and the eval/fine-tune cost matches. Symbols are
    taken in sorted id order, biased toward documented functions first.
    """
    tree = Path(tree) if tree is not None else _DEFAULT_TREE
    if not tree.is_dir():
        raise FileNotFoundError(
            f"source tree not found: {tree}. agent-harness must be on disk to "
            "regenerate; the committed fixture is the offline source of truth.")

    files = _iter_py_files(tree)
    mod_to_file = {_module_name(tree, f, _PKG_ROOT): _rel(tree, f) for f in files}

    # pass 1 — collect candidate chunks + a symbol-name -> ids index
    candidates: list[dict] = []
    sym_by_name: dict[str, set[str]] = {}
    file_imports: dict[str, set[str]] = {}
    for f in files:
        rel = _rel(tree, f)
        try:
            mod = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (SyntaxError, UnicodeDecodeError):
            continue
        cur_mod = _module_name(tree, f, _PKG_ROOT)
        # internal file imports for this module (for co-locating import edges)
        imps: set[str] = set()
        for imp in ast.walk(mod):
            if isinstance(imp, ast.ImportFrom) and imp.module and not imp.level:
                dst = _resolve_module(imp.module, mod_to_file)
                if dst and dst != rel:
                    imps.add(dst)
            elif isinstance(imp, ast.ImportFrom) and imp.level:
                base = cur_mod.rsplit(".", imp.level)[0] if "." in cur_mod else _PKG_ROOT
                dst = _resolve_module(f"{base}.{imp.module}" if imp.module else base,
                                      mod_to_file)
                if dst and dst != rel:
                    imps.add(dst)
        file_imports[rel] = imps

        for qual, node in _top_level_symbols(mod):
            cid = f"{rel}::{qual}"
            short = qual.split(".")[-1]
            candidates.append({
                "id": cid, "symbol": short, "file": rel,
                "signature": _signature(node),
                "docstring": _docstring(node),
                "body": _body_summary(node),
                "_calls": [c for c in (_call_name(s.func) for s in ast.walk(node)
                                       if isinstance(s, ast.Call)) if c],
                "_documented": bool(_docstring(node)),
            })
            sym_by_name.setdefault(short, set()).add(cid)

    # select chunks: documented first, then by id, capped — keeps a real,
    # comparable-size corpus and ensures most chunks carry real semantics.
    candidates.sort(key=lambda c: (not c["_documented"], c["id"]))
    selected = candidates[:max_chunks]
    chunk_ids = {c["id"] for c in selected}
    by_file: dict[str, list[str]] = {}
    for c in selected:
        by_file.setdefault(c["file"], []).append(c["id"])

    # pass 2 — build edges, FILTERED to references inside the corpus
    chunks: list[dict] = []
    for c in selected:
        # call edges -> unambiguous internal symbol that is also a chunk
        call_targets: list[str] = []
        for name in c["_calls"]:
            cands = {cid for cid in sym_by_name.get(name, set())
                     if cid in chunk_ids and cid != c["id"]}
            if len(cands) == 1:
                t = next(iter(cands))
                if t not in call_targets:
                    call_targets.append(t)
        # import edges: chunks in files this file imports (cap for balance)
        import_targets: list[str] = []
        for dst_file in sorted(file_imports.get(c["file"], set())):
            for cid in by_file.get(dst_file, [])[:2]:
                if cid != c["id"]:
                    import_targets.append(cid)
        # co_edit: other chunks in the SAME file (real same-file locality)
        co_edits = sorted(cid for cid in by_file.get(c["file"], [])
                          if cid != c["id"])

        chunks.append({
            "id": c["id"], "symbol": c["symbol"], "file": c["file"],
            "signature": c["signature"],
            "docstring": c["docstring"] or f"{c['symbol']} in {c['file']}.",
            "body": c["body"],
            "imports": sorted(set(call_targets + import_targets))[:6],
            "co_edits": co_edits[:8],
        })

    return {"source_tree": _safe_tree_label(tree), "pkg_root": _PKG_ROOT,
            "chunks": chunks}


def _safe_tree_label(tree: Path) -> str:
    try:
        return "~/" + tree.relative_to(Path.home()).as_posix()
    except ValueError:
        return tree.name


def write_corpus(tree: Path | None = None, out_path: Path | None = None) -> dict:
    data = build_corpus(tree)
    out = Path(out_path) if out_path is not None else paths.agent_harness_codebase_json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"chunks": len(data["chunks"]),
            "documented": sum(1 for c in data["chunks"]
                              if c["docstring"] and "in " not in c["docstring"][:4])}


def load_corpus() -> list[dict]:
    p = paths.agent_harness_codebase_json()
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; regenerate with `uv run python -c \"from "
            "course_lab.agent_harness_corpus import write_corpus; write_corpus()\"`")
    data = json.loads(p.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    ids = {c["id"] for c in chunks}
    for c in chunks:
        for ref in c["imports"] + c["co_edits"]:
            if ref not in ids:
                raise ValueError(f"chunk {c['id']} references unknown id {ref}")
    return chunks


if __name__ == "__main__":  # pragma: no cover
    print(write_corpus())
