"""SGF 构建 —— 移植自 TS `sgf.ts`。"""
from __future__ import annotations

from .types import DetectedStone

ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _coord(col: int, row: int) -> str:
    return ALPHA[col] + ALPHA[row]


def build_sgf(board_size: int, stones: list[DetectedStone]) -> str:
    """构建静态局面（AM/AW 属性，无走子）。对齐 TS buildSGF。"""
    black = [s for s in stones if s.color == "black"]
    white = [s for s in stones if s.color == "white"]

    props = f"GM[1]FF[4]SZ[{board_size}]AP[Kaya Board Recognition]"
    if black:
        props += "AB" + "".join(f"[{_coord(s.x, s.y)}]" for s in black)
    if white:
        props += "AW" + "".join(f"[{_coord(s.x, s.y)}]" for s in white)
    return f"(;{props}\n)"
