"""Parse a REAL Python source tree into a code graph (import / call / co_edit).

Module 3's codebase arm (arm A) used to run on a hand-built toy graph
(``seed_coding_agent``). This parser replaces that with a genuine source tree:
it walks the real package with Python's ``ast`` for ``import`` and ``call``
edges and ``git log`` for ``co_edit`` edges, then derives labeled multi-hop
eval tasks from the actual call/import structure (so the eval is real, not
hand-written for invented symbols).

Default target: ``~/personal/agent-harness/src/agent_harness`` — a real coding
agent we built (26 subpackages, ~200 modules, ~390 internal imports). The output
is a committed fixture (``course_lab/data/agent_harness_code_graph.json``) so the
notebook/tests run offline and deterministically, mirroring the SCM fixture
convention.

Node id namespace (matches the rest of Module 3):
  * ``file:<relpath>``           — a module, e.g. ``file:core/agent_loop.py``
  * ``sym:<relpath>::<qualname>`` — a top-level function/method/class

Edge kinds: ``import`` (file->file), ``call`` (sym->sym), ``co_edit`` (file<->file).
"""
from __future__ import annotations

import ast
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from course_lab import paths

_DEFAULT_TREE = Path.home() / "personal" / "agent-harness" / "src" / "agent_harness"
_PKG_ROOT_NAME = "agent_harness"


# --- module/symbol discovery -----------------------------------------------

def _iter_py_files(tree: Path) -> list[Path]:
    files = [
        p for p in sorted(tree.rglob("*.py"))
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]
    return files


def _module_name(tree: Path, path: Path, pkg_root: str) -> str:
    """Dotted module name as imports would reference it (pkg_root.sub.mod)."""
    rel = path.relative_to(tree).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([pkg_root, *parts]) if parts else pkg_root


def _rel(tree: Path, path: Path) -> str:
    return path.relative_to(tree).as_posix()


def _top_level_symbols(tree_node: ast.AST) -> list[tuple[str, ast.AST]]:
    """Top-level functions, async functions, classes, and one level of methods."""
    out: list[tuple[str, ast.AST]] = []
    for node in tree_node.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            out.append((node.name, node))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((f"{node.name}.{sub.name}", sub))
    return out


# --- parse ------------------------------------------------------------------

def parse_tree(tree: Path | None = None, *, pkg_root: str = _PKG_ROOT_NAME) -> dict:
    """Parse a real Python tree into {nodes, edges, eval_tasks}. Deterministic."""
    tree = Path(tree) if tree is not None else _DEFAULT_TREE
    if not tree.is_dir():
        raise FileNotFoundError(
            f"source tree not found: {tree}. Point code_graph_parse.parse_tree() "
            "at a real Python package directory.")

    files = _iter_py_files(tree)
    # module dotted-name -> file relpath, for resolving internal imports
    mod_to_file = {_module_name(tree, f, pkg_root): _rel(tree, f) for f in files}
    file_set = {_rel(tree, f) for f in files}

    nodes: dict[str, dict] = {}
    import_edges: set[tuple[str, str]] = set()
    call_edges: set[tuple[str, str]] = set()
    # symbol short-name -> set of full sym ids (for resolving call targets)
    sym_by_name: dict[str, set[str]] = defaultdict(set)
    # per-file parsed module + symbol list (two-pass: collect, then resolve calls)
    file_syms: dict[str, list[tuple[str, ast.AST]]] = {}

    # pass 1 — nodes + imports + symbol table
    for f in files:
        rel = _rel(tree, f)
        nodes[f"file:{rel}"] = {"id": f"file:{rel}", "kind": "file",
                                "text": f"Module {rel} in {pkg_root}."}
        try:
            mod = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (SyntaxError, UnicodeDecodeError):
            file_syms[rel] = []
            continue
        cur_mod = _module_name(tree, f, pkg_root)
        syms = _top_level_symbols(mod)
        file_syms[rel] = syms
        for qual, _node in syms:
            sid = f"sym:{rel}::{qual}"
            short = qual.split(".")[-1]
            nodes[sid] = {"id": sid, "kind": "sym",
                          "text": f"{short} in {rel} ({pkg_root})."}
            sym_by_name[short].add(sid)

        # imports -> resolve internal targets to file: nodes
        for imp in ast.walk(mod):
            targets: list[str] = []
            if isinstance(imp, ast.Import):
                targets = [a.name for a in imp.names]
            elif isinstance(imp, ast.ImportFrom):
                if imp.level:  # relative import — resolve against current package
                    base = cur_mod.rsplit(".", imp.level)[0] if "." in cur_mod else pkg_root
                    mod_part = f"{base}.{imp.module}" if imp.module else base
                    targets = [mod_part] + [f"{mod_part}.{a.name}" for a in imp.names]
                elif imp.module:
                    targets = [imp.module] + [f"{imp.module}.{a.name}" for a in imp.names]
            for t in targets:
                # match the longest known module prefix to a file
                dst = _resolve_module(t, mod_to_file)
                if dst and dst != rel:
                    import_edges.add((f"file:{rel}", f"file:{dst}"))

    # pass 2 — calls (resolve callee short-name to a unique internal symbol)
    for rel, syms in file_syms.items():
        for qual, node in syms:
            caller = f"sym:{rel}::{qual}"
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    callee_name = _call_name(sub.func)
                    if not callee_name:
                        continue
                    cands = sym_by_name.get(callee_name, set())
                    # only keep edges to an UNAMBIGUOUS internal symbol, and not self
                    cands = {c for c in cands if c != caller}
                    if len(cands) == 1:
                        call_edges.add((caller, next(iter(cands))))

    co_edit_edges = _co_edit_edges(tree, file_set)

    edges = (
        [{"src": s, "dst": d, "kind": "import"} for s, d in sorted(import_edges)]
        + [{"src": s, "dst": d, "kind": "call"} for s, d in sorted(call_edges)]
        + [{"src": s, "dst": d, "kind": "co_edit"} for s, d in sorted(co_edit_edges)]
    )
    eval_tasks = _derive_eval_tasks(nodes, call_edges, import_edges)

    # store a home-relative source path (portable, no absolute /home/<user> leak)
    try:
        tree_label = "~/" + tree.relative_to(Path.home()).as_posix()
    except ValueError:
        tree_label = tree.name

    return {
        "source_tree": tree_label,
        "pkg_root": pkg_root,
        "n_files": len(file_set),
        "nodes": [nodes[k] for k in sorted(nodes)],
        "edges": edges,
        "eval_tasks": eval_tasks,
    }


def _resolve_module(dotted: str, mod_to_file: dict[str, str]) -> str | None:
    """Map a dotted import target to a file relpath by longest known prefix."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in mod_to_file:
            return mod_to_file[cand]
    return None


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# --- co_edit from git -------------------------------------------------------

def _co_edit_edges(tree: Path, file_set: set[str],
                   *, max_commits: int = 1000) -> set[tuple[str, str]]:
    """Files changed together in the same commit (real git history)."""
    repo = _git_root(tree)
    if repo is None:
        return set()
    rel_prefix = tree.relative_to(repo).as_posix()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", f"-n{max_commits}",
             "--name-only", "--pretty=format:%H", "--", rel_prefix],
            capture_output=True, text=True, timeout=120, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return set()

    edges: set[tuple[str, str]] = set()
    commit_files: list[str] = []

    def _flush(group: list[str]) -> None:
        rels = sorted({p for p in group if p in file_set})
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                edges.add((f"file:{rels[i]}", f"file:{rels[j]}"))

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if "/" not in line and len(line) == 40:  # commit sha line
            _flush(commit_files)
            commit_files = []
        elif line.endswith(".py"):
            # git path is repo-relative; strip the tree prefix to match file_set
            if line.startswith(rel_prefix + "/"):
                commit_files.append(line[len(rel_prefix) + 1:])
    _flush(commit_files)
    return edges


def _git_root(path: Path) -> Path | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return Path(out) if out else None
    except (subprocess.SubprocessError, OSError):
        return None


# --- derived eval tasks (real multi-hop, not hand-written) ------------------

def _derive_eval_tasks(nodes: dict, call_edges: set, import_edges: set,
                       *, max_per_type: int = 12) -> list[dict]:
    """Build labeled (query, positive, type) tasks FROM the real graph.

    multi_hop: callee of a real call edge — named via the CALLER, so the positive
    (callee) is not lexically in the query and only graph propagation reaches it.
    lexical : direct "where is X defined" hits where structure does no work.
    Deterministic (sorted inputs, fixed phrasings).
    """
    def short(sid: str) -> str:
        return sid.split("::")[-1].split(".")[-1]

    # one distinct callee per caller (first by sorted order), and require the
    # callee short-name to differ from the caller's so the positive is genuinely
    # not the query subject — a real multi-hop reach, not a lexical echo.
    tasks: list[dict] = []
    used_callers: set[str] = set()
    for caller, callee in sorted(call_edges):
        if len(tasks) >= max_per_type:
            break
        cn, ce = short(caller), short(callee)
        if cn == ce or caller in used_callers:
            continue
        used_callers.add(caller)
        tasks.append({
            "query": f"When {cn} runs, which internal function does it call?",
            "positive": callee, "type": "multi_hop"})

    lex: list[dict] = []
    for sid in sorted(nodes):
        if len(lex) >= max_per_type:
            break
        if sid.startswith("sym:"):
            lex.append({"query": f"Where is {short(sid)} defined?",
                        "positive": sid, "type": "lexical"})
    return tasks + lex


# --- fixture IO -------------------------------------------------------------

def write_fixture(tree: Path | None = None, out_path: Path | None = None) -> dict:
    """Parse the tree and write the committed fixture. Returns a small summary."""
    data = parse_tree(tree)
    out = Path(out_path) if out_path is not None else paths.agent_harness_code_graph_json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    kinds: dict[str, int] = defaultdict(int)
    for e in data["edges"]:
        kinds[e["kind"]] += 1
    return {"n_files": data["n_files"], "n_nodes": len(data["nodes"]),
            "edges": dict(kinds), "n_eval_tasks": len(data["eval_tasks"])}


def load_code_graph() -> dict:
    p = paths.agent_harness_code_graph_json()
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; regenerate with `uv run python -c \"from "
            "course_lab.code_graph_parse import write_fixture; print(write_fixture())\"`")
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    print(write_fixture())
