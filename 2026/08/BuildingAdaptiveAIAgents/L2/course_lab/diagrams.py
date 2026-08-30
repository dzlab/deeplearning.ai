"""course_lab/diagrams.py: the house style for every course diagram.

Each module renders its graphs through :func:`house_style_mermaid` so the whole
course reads as one set: horizontal flow, calm greyscale nodes, and a single
Oracle Red accent reserved for the loop-back edge (the arc that makes an agent
loop a loop). Keeping the palette and the builder in one place is what stops the
four modules' diagrams from quietly drifting apart.

Render to an image with :func:`render_png`; it falls back to ``None`` when there
is no renderer or no network, so callers can print the Mermaid text instead.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

# ── House palette ─────────────────────────────────────────────────────────────
# Greyscale carries the structure; Oracle Red is spent only on the loop arc.
ORACLE_RED = "#C74634"  # the one accent: loop-back edges only
NODE_FILL = "#F2F2F2"  # process-node fill
NODE_BORDER = "#8A8A8A"  # process-node border and default arrows
ARROW_GREY = "#8A8A8A"  # every non-loop edge
TERMINAL_FILL = "#555555"  # START / END pills
TEXT_DARK = "#1A1A1A"  # label text on light nodes
TEXT_LIGHT = "#FFFFFF"  # label text on the dark terminal pills

# A node is (id, label, kind) with kind in {"process", "terminal"}.
Node = Tuple[str, str, str]
# An edge is (src, dst, label, kind) with kind in {"normal", "loop"}; label "" hides it.
Edge = Tuple[str, str, str, str]


def house_style_mermaid(
    nodes: Iterable[Node],
    edges: Sequence[Edge],
    *,
    direction: str = "LR",
) -> str:
    """Build a house-style Mermaid flowchart string.

    Args:
        nodes: ``(node_id, label, kind)`` triples. ``kind`` is ``"process"``
            (rounded grey box) or ``"terminal"`` (dark pill, for START/END).
            ``node_id`` must be Mermaid-safe: avoid the reserved word ``end``.
        edges: ``(src, dst, label, kind)`` tuples in draw order. ``label`` ""
            renders a bare arrow. ``kind`` ``"loop"`` paints the edge Oracle Red;
            ``"normal"`` leaves it grey.
        direction: Mermaid flow direction; defaults to ``"LR"`` (horizontal) so
            every diagram in the course shares one orientation.

    Returns:
        A Mermaid ``graph`` string ready for :func:`render_png` or a notebook.

    Example:
        >>> nodes = [("a", "START", "terminal"), ("b", "step", "process")]
        >>> edges = [("a", "b", "", "normal")]
        >>> print(house_style_mermaid(nodes, edges))  # doctest: +ELLIPSIS
        graph LR...
    """
    lines = [f"graph {direction}"]

    # ── Nodes: rounded box for work, dark pill for the start/stop terminals ────
    for node_id, label, kind in nodes:
        if kind == "terminal":
            lines.append(f'    {node_id}(["{label}"]):::terminal')
        elif kind == "process":
            lines.append(f'    {node_id}("{label}"):::proc')
        else:
            raise ValueError(
                f"unknown node kind {kind!r} for {node_id!r}; "
                "expected 'process' or 'terminal'"
            )

    # ── Edges: track which ones are loops so we can colour them by index ───────
    loop_indices = []
    for index, (src, dst, label, kind) in enumerate(edges):
        if kind not in ("normal", "loop"):
            raise ValueError(
                f"unknown edge kind {kind!r} for {src!r}->{dst!r}; "
                "expected 'normal' or 'loop'"
            )
        if label:
            lines.append(f"    {src} -->|{label}| {dst}")
        else:
            lines.append(f"    {src} --> {dst}")
        if kind == "loop":
            loop_indices.append(index)

    # ── House style: greyscale by default, Oracle Red reserved for the loop ────
    lines.append(f"    classDef proc fill:{NODE_FILL},stroke:{NODE_BORDER},color:{TEXT_DARK};")
    lines.append(f"    classDef terminal fill:{TERMINAL_FILL},stroke:{TERMINAL_FILL},color:{TEXT_LIGHT};")
    lines.append(f"    linkStyle default stroke:{ARROW_GREY},stroke-width:1.5px;")
    if loop_indices:
        idx = ",".join(str(i) for i in loop_indices)
        lines.append(f"    linkStyle {idx} stroke:{ORACLE_RED},stroke-width:2px;")

    return "\n".join(lines)


def render_png(mermaid_syntax: str) -> Optional[bytes]:
    """Render a Mermaid string to PNG bytes, or ``None`` if that is not possible.

    Rendering goes through ``mermaid.ink`` (via LangChain), so it needs a
    network connection. Any failure (offline, renderer missing) returns ``None``
    so the caller can fall back to printing the Mermaid text.

    Args:
        mermaid_syntax: A Mermaid ``graph`` string, e.g. from
            :func:`house_style_mermaid`.

    Returns:
        PNG image bytes, or ``None`` when no renderer is reachable.
    """
    try:
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png

        return draw_mermaid_png(mermaid_syntax)
    except Exception:
        return None
