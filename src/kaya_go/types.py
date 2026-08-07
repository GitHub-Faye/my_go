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


@dataclass
class MokuRawDetection:
    """Moku 原始检出（原图坐标，用于角点重映射）。"""

    cx: float
    cy: float
    class_id: int
    score: float


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

    # 序列化为 JSON 时的辅助方法，见 recognition.py 顶层的 serializer
    def to_dict(self) -> dict:
        return {
            "boardSize": self.board_size,
            "stones": [{"x": s.x, "y": s.y, "color": s.color} for s in self.stones],
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
