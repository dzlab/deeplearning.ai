"""course_lab/pageindex.py — PageIndex hierarchical document tree (pure-Python).

A PageIndex tree is a vectorless table-of-contents index: each node carries a
title, an LLM-style summary, and a source page range; children nest subsections.
This module loads the committed real Apple FY2024 10-K tree, flattens it to
retrievable nodes, derives parent->child graph edges (kind "contains"), and
supports a *surgical append* of one node — the continual-learning move that
touches only the new node and its single parent edge.

Node ids are namespaced ``pi:<node_id>``; the synthetic root is ``pi:root`` so
the document itself is a node that every Part hangs from.
"""
from __future__ import annotations

import copy
import json

from course_lab import paths

ROOT_ID = "pi:root"


def load_tree() -> dict:
    """Load the committed PageIndex tree fixture."""
    return json.loads(paths.pageindex_apple_10k_json().read_text(encoding="utf-8"))


def load_pages() -> dict[str, str]:
    """Load the committed ``{page: text}`` snippets for the 10-K's financial pages.

    A small set of real pages (Apple's actual reported figures) so the
    navigation demo can pull a concrete answer deterministically and offline.
    """
    raw = json.loads(paths.pageindex_apple_10k_pages_json().read_text(encoding="utf-8"))
    return dict(raw.get("pages", {}))


def page_text(page: int, pages: dict[str, str] | None = None) -> str:
    """The committed text for one page (empty string if not in the snippet set)."""
    pages = load_pages() if pages is None else pages
    return pages.get(str(page), "")


def node_pages_text(node: dict, pages: dict[str, str] | None = None) -> str:
    """Concatenate the committed page text spanning ``node``'s page range.

    ``node`` is a flattened node (has ``pages: [start, end]``). Only pages that
    exist in the snippet set contribute, so callers can detect a miss by an
    empty result.
    """
    pages = load_pages() if pages is None else pages
    span = node.get("pages") or []
    if len(span) != 2:
        return ""
    start, end = int(span[0]), int(span[1])
    chunks = [pages[str(p)] for p in range(start, end + 1) if str(p) in pages]
    return "\n".join(chunks)


def node_text(node: dict) -> str:
    """The text that gets embedded / stored for a node: 'Title — summary'."""
    title = str(node.get("title", "")).strip()
    summary = str(node.get("summary", "")).strip()
    return f"{title} — {summary}" if summary else title


def flatten(tree: dict) -> list[dict]:
    """Flatten to a list of ``{id, title, summary, pages, parent}`` (root excluded).

    Top-level children have parent ``pi:root``; deeper nodes have their real
    parent. ``id`` is ``pi:<node_id>``.
    """
    out: list[dict] = []

    def walk(node: dict, parent_id: str) -> None:
        nid = "pi:" + node["node_id"]
        out.append({
            "id": nid,
            "title": node.get("title", ""),
            "summary": node.get("summary", ""),
            "pages": node.get("pages"),
            "parent": parent_id,
        })
        for child in node.get("children", []):
            walk(child, nid)

    for top in tree.get("children", []):
        walk(top, ROOT_ID)
    return out


def to_graph_edges(tree: dict) -> list[tuple]:
    """Parent->child ``contains`` edges, including ``pi:root`` -> each Part."""
    return [(n["parent"], n["id"], "contains") for n in flatten(tree)]


def append_node(tree: dict, parent_node_id: str, new_node: dict) -> dict:
    """Return a NEW tree with ``new_node`` appended under ``parent_node_id``.

    Pure: the input tree is deep-copied, so callers can compare before/after.
    ``parent_node_id`` is the bare node_id (e.g. ``"i7"``), not the ``pi:`` id.
    Raises KeyError if the parent is not found.
    """
    t = copy.deepcopy(tree)

    def find(node: dict):
        if node.get("node_id") == parent_node_id:
            return node
        for child in node.get("children", []):
            hit = find(child)
            if hit is not None:
                return hit
        return None

    # the synthetic root has no node_id; allow appending a new Part to it
    parent = None
    if parent_node_id == "root":
        parent = t
    else:
        for top in t.get("children", []):
            parent = find(top)
            if parent is not None:
                break
    if parent is None:
        raise KeyError(f"parent node_id {parent_node_id!r} not found")
    parent.setdefault("children", []).append(copy.deepcopy(new_node))
    return t
