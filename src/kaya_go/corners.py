"""角点排序与四边形检测 —— 移植自 TS `corners.ts`。

包含了 orderCorners / spreadCollapsedCorners / insetImageCorners 等纯几何逻辑。
"""
from __future__ import annotations

import math

from .types import BoardCorners, Point

# ── 角点排序 ────────────────────────────────────────────────────────────────


def order_corners(points: list[Point]) -> BoardCorners:
    """按 TL → TR → BR → BL 顺序排序（沿中心逆时针）。对齐 TS orderCorners。"""
    if len(points) != 4:
        raise ValueError(f"需要 4 个角点，得到 {len(points)}")
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4
    # 按极角升序排（从 Tr 手方向逆时针）
    sorted_pts = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    # 以 x+y 最小的点为左上角，顺（极角增序）取 4 个
    tl_idx = min(range(4), key=lambda i: sorted_pts[i][0] + sorted_pts[i][1])
    return tuple(sorted_pts[(tl_idx + i) % 4] for i in range(4))  # type: ignore[return-value]


def inset_image_corners(w: int, h: int, fraction: float) -> BoardCorners:
    """沿图像边缘向内缩进 fraction 比例作为兜底角点。"""
    m = min(w, h) * fraction
    return (
        (m, m),
        (w - 1 - m, m),
        (w - 1 - m, h - 1 - m),
        (m, h - 1 - m),
    )


def are_corners_degenerate(
    corners: BoardCorners, img_width: int, img_height: int, min_fraction: float = 0.02
) -> bool:
    """4 角点是否聚拢退化（包围盒面积 < 图像面积 * min_fraction）。"""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return bbox_area < img_width * img_height * min_fraction


def spread_collapsed_corners(
    corners: BoardCorners, width: int, height: int, min_dist_fraction: float = 0.05
) -> tuple[BoardCorners, bool]:
    """检测 2+ 角点塌缩（距离 < 图像对角线 * fraction），塌缩则发散到默认插入点。"""
    diag = math.hypot(width, height)
    min_dist = diag * min_dist_fraction

    collapsed = False
    for i in range(4):
        for j in range(i + 1, 4):
            d = math.hypot(corners[i][0] - corners[j][0], corners[i][1] - corners[j][1])
            if d < min_dist:
                collapsed = True
                break
        if collapsed:
            break

    if not collapsed:
        return corners, False

    m = min(width, height) * 0.05
    fallback: BoardCorners = (
        (m, m),
        (width - 1 - m, m),
        (width - 1 - m, height - 1 - m),
        (m, height - 1 - m),
    )
    return fallback, True


def expand_corners(
    corners: BoardCorners, img_width: int, img_height: int, margin: float
) -> BoardCorners:
    """以质心为中心向外扩展 margin 比例。"""
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    out: list[Point] = []
    for px, py in corners:
        dx = px - cx
        dy = py - cy
        nx = min(max(0, px + dx * margin), img_width - 1)
        ny = min(max(0, py + dy * margin), img_height - 1)
        out.append((nx, ny))
    return tuple(out)  # type: ignore[return-value]
