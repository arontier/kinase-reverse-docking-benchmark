"""
Stamps panel letters (a)(b)(c)... outside the axes, at the top left.

Why outside: placed inside the axes (say transAxes 0.02, 0.96) they collide with legends, curves
and tick labels. The manuscript captions already use the lower-case parenthesised form
("Panel (a) shows..."), so the style matches that.
"""
from __future__ import annotations

import string


def stamp(axes, dx: float = -0.09, dy: float = 1.06, size: float = 13) -> None:
    """axes 순서대로 (a), (b), (c) … 를 찍는다. colorbar 축은 넘기지 말 것."""
    try:
        seq = list(axes.ravel())          # numpy array of Axes
    except AttributeError:
        seq = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    for ax, ch in zip(seq, string.ascii_lowercase):
        ax.text(dx, dy, f"({ch})", transform=ax.transAxes,
                fontsize=size, fontweight="bold", va="bottom", ha="left")
