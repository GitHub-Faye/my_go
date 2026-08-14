"""核心类型定义 —— 移植自 TS `packages/board-recognition/src/types.ts`。

为保证与浏览器端识别结果一致，字段命名尽量沿用 TS 版本。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Point: (x, y)
Point = tuple[float, float]
# BoardCorners: [TL, TR, BR, BL]
BoardCorners = tuple[Point, Point, Point, Point]

StoneColor = Literal["black", "white"]


@dataclass
class DetectedStone:
    """x=列(0-indexed)，y=行(0-indexed)。对齐 TS DetectedStone。"""

    x: int
    y: int
    color: StoneColor

    def sign(self) -> int:
        """映射为 deadstones 的 Sign: 1=黑, -1=白。"""
        return 1 if self.color == "black" else -1


@dataclass
class MokuRawDetection:
    """Moku 原始检出（原图坐标，用于角点重映射）。"""

    cx: float
    cy: float
    class_id: int
    score: float


def _build_sign_map(board_size: int, stones: list[DetectedStone]) -> list[list[int]]:
    """由石头构造棋盘矩阵：signMap[y][x] ∈ {1=黑, -1=白, 0=空}。

    DetectedStone.x=列, y=行,故 signMap[y]=行,x=列。行列超出棋盘时跳过
    （识别偶发散点）,保证输出恒为 board_size×board_size。
    """
    sign_map = [[0] * board_size for _ in range(board_size)]
    for s in stones:
        if 0 <= s.y < board_size and 0 <= s.x < board_size:
            sign_map[s.y][s.x] = s.sign()
    return sign_map


@dataclass
class RecognitionResult:
    board_size: int
    stones: list[DetectedStone]
    corners: BoardCorners
    corners_detected: bool
    sgf: str
    estimated_grid_corners: BoardCorners | None = None
    moku_raw_corners: BoardCorners | None = None
    moku_corner_count: int | None = None
    # 不返回 warpedImage（服务端无需回传大图）

    def build_sign_map(self) -> list[list[int]]:
        return _build_sign_map(self.board_size, self.stones)

    # 序列化为 JSON 时的辅助方法，见 main.recognize 顶层的 serializer
    def to_dict(self) -> dict:
        """序列化为 JSON 辅助：保持既有 /api/v1/recognize 响应契约不变。"""
        return {
            "boardSize": self.board_size,
            "stones": [{"x": s.x, "y": s.y, "color": s.color} for s in self.stones],
            "signMap": self.build_sign_map(),
            "corners": [list(pts) for pts in self.corners],
            "cornersDetected": self.corners_detected,
            "sgf": self.sgf,
            "estimatedGridCorners": (
                [list(pts) for pts in self.estimated_grid_corners]
                if self.estimated_grid_corners
                else None
            ),
            "mokuRawCorners": (
                [list(pts) for pts in self.moku_raw_corners]
                if self.moku_raw_corners
                else None
            ),
            "mokuCornerCount": self.moku_corner_count,
        }


# 下一步该谁走 —— 黑先原则下的静态推断。供 AI 分析（MCTS 需 nextToPlay）使用。
NextToPlay = Literal["B", "W", "unknown"]


def derive_next_to_play(sign_map: list[list[int]]) -> NextToPlay:
    """由棋盘快照推断该谁走（围棋黑先约定）。

    - 黑白子数相同     → 该黑走（'B'）
    - 白比黑多一子     → 该白走（'W'）
    - 其余（子数差 >1，通常伴随提子/打吃，无法从快照唯一确定）→ 'unknown'

    纯数量启发式：子数差恰好 0/1 是正常手数推进，能直接推出；一旦出现提子
    （被抓棋子被移除导致子数失衡），仅凭快照无法反推手数，故保守返回 unknown，
    由调用方（用户体验）人工指定。
    """
    black = sum(1 for row in sign_map for v in row if v == 1)
    white = sum(1 for row in sign_map for v in row if v == -1)
    if black == white:
        return "B"
    if white == black + 1:
        return "W"
    return "unknown"


def derive_next_to_play(sign_map: list[list[int]]) -> NextToPlay:
    """由棋盘快照推断该谁走（围棋黑先约定）。

    - 黑白子数相同     → 该黑走（'B'）
    - 白比黑多一子     → 该白走（'W'）
    - 其余（子数差 >1，通常伴随提子/打吃，无法从快照唯一确定）→ 'unknown'

    纯数量启发式：子数差恰好 0/1 是正常手数推进，能直接推出；一旦出现提子
    （被抓棋子被移除导致子数失衡），仅凭快照无法反推手数，故保守返回 unknown，
    由调用方（用户体验）人工指定。
    """
    black = sum(1 for row in sign_map for v in row if v == 1)
    white = sum(1 for row in sign_map for v in row if v == -1)
    if black == white:
        return "B"
    if white == black + 1:
        return "W"
    return "unknown"
