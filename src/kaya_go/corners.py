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


def _orth_unit(a: Point, b: Point, img_w: int, img_h: int) -> Point:
    """边 (a→b) 的正交单位向量，取使中点更靠近图片中心的那一侧。

    top/bottom 边返回"朝上下"、left/right 边返回"朝左右"的方向，
    供后续把塌缩的角沿该方向外扩到覆盖大片的区域。
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    to_center = ((img_w - 1) / 2 - mid[0], (img_h - 1) / 2 - mid[1])
    for o in ((-dy, dx), (dy, -dx)):
        if o[0] * to_center[0] + o[1] * to_center[1] > 0:
            d = math.hypot(*o)
            return (o[0] / d, o[1] / d) if d > 0 else (0.0, 1.0)
    return (0.0, 1.0)


# 塌缩判定：最短边 < 最长边 * 此比例，或 < 图像对角线 * 此比例
COLLAPSE_EDGE_RATIO = 0.4
COLLAPSE_DIAG_FRAC = 0.08


def adjust_degenerate_corners(
    corners: BoardCorners, img_width: int, img_height: int
) -> tuple[BoardCorners, bool]:
    """修复"塌缩到图片一边"的角点，使四角覆盖大片图片内容。

    问题场景：moku 检出的 4 个角里，某条边明显过短——它的两个端点在照片里
    挤到了图像一角（如示例中 TL/TR 同时塌缩到最上方），无法构成覆盖大片
    内容的近似正方形。此时保留该短边的**对边**（= 仍在合理位置、能作为覆盖
    大片的可靠锚定的两点），沿对边的正交方向把塌缩的两个角重新推导到图片
    中央侧，得到覆盖大片的近似矩形。

    判定规则（比 spread_collapsed 更灵敏，且定向修补而非整盘回退）：
      - 最短边 < 最长边 * COLLAPSE_EDGE_RATIO，或
      - 最短边 < 图像对角线 * COLLAPSE_DIAG_FRAC
    时视为存在塌缩边，返回 (修补后四角, True)；否则原样返回 (corners, False)。

    返回顺序仍为 TL → TR → BR → BL。
    """
    tl, tr, br, bl = corners
    top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    right = math.hypot(br[0] - tr[0], br[1] - tr[1])
    bottom = math.hypot(bl[0] - br[0], bl[1] - br[1])
    left = math.hypot(tl[0] - bl[0], tl[1] - bl[1])

    diag = math.hypot(img_width, img_height)
    shortest = min(top, right, bottom, left)
    longest = max(top, right, bottom, left)
    if shortest >= longest * COLLAPSE_EDGE_RATIO and shortest >= diag * COLLAPSE_DIAG_FRAC:
        return corners, False

    # 定位最短边，选锚边（对边）的两个端点；orth 指向图片中央一侧。
    if shortest == top:
        anchor_a, anchor_b = bl, br
    elif shortest == bottom:
        anchor_a, anchor_b = tl, tr
    elif shortest == left:
        anchor_a, anchor_b = tr, br
    else:  # right
        anchor_a, anchor_b = tl, bl

    orth = _orth_unit(anchor_a, anchor_b, img_width, img_height)
    # 塌缩端的合理推导距离：短边所在对的可信边长最大值（兜底用最长边），
    # 保证修补后的四边形仍覆盖大致同样大小的内容。
    if shortest in (top, bottom):
        height = max(left, right, longest)
    else:
        height = max(top, bottom, longest)

    # 锚边两端在合理位置，各沿 orth 外扩 height → 得到塌缩的两个角
    extended_a = (anchor_a[0] + orth[0] * height, anchor_a[1] + orth[1] * height)
    extended_b = (anchor_b[0] + orth[0] * height, anchor_b[1] + orth[1] * height)

    def _clamp(p: Point) -> Point:
        return (
            min(max(0.0, p[0]), img_width - 1),
            min(max(0.0, p[1]), img_height - 1),
        )

    # 重组为 TL/TR/BR/BL：锚边保留，锚边两端各自对应塌缩端
    if shortest == top:
        out = (extended_a, extended_b, br, bl)
    elif shortest == bottom:
        out = (tl, tr, extended_b, extended_a)
    elif shortest == left:
        out = (extended_a, tr, br, extended_b)
    else:  # right
        out = (tl, extended_a, extended_b, bl)

    out = tuple(_clamp(p) for p in out)  # type: ignore[assignment]
    return out, True


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


# ── 近似四边形判定 + 可靠锚点选择 ──────────────────────────────────────────

# 可靠锚点构成「近似三角形」的下限：面积 ≥ 图像面积 * 此比例，且
# 三边中任意一边不短于对称退化阈值（COLLAPSE_DIAG_FRAC）
MIN_TRI_AREA_FRACTION = 0.005
# 可靠锚点退化为 2 点时，两角距离需不短于图像对角线 * 此比例才算「足够长的对角」
MIN_ANCHOR_DIAG_FRACTION = 0.4


def _tri_plausible(a: Point, b: Point, c: Point, W: int, H: int) -> bool:
    """三点是否构成合理的非退化三角形（面积足够大、三边不塌缩）。"""
    diag = math.hypot(W, H)
    e1 = math.hypot(a[0] - b[0], a[1] - b[1])
    e2 = math.hypot(b[0] - c[0], b[1] - c[1])
    e3 = math.hypot(c[0] - a[0], c[1] - a[1])
    if min(e1, e2, e3) < diag * COLLAPSE_DIAG_FRAC:
        return False
    area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    return area > (W * H) * MIN_TRI_AREA_FRACTION


def quad_is_plausible(corners: BoardCorners, W: int, H: int) -> bool:
    """4 个有序角点是否构成近似四边形 ——「任三角近似」检验。

    对 4 点中任取 3 点构成的每个三角形都要近似成立；只要存在一个
    退化/塌缩三角形（说明里头有坏点），整体就不算近似四边形，应触发
    用可靠锚点重建。比单纯看边长短更宽松，能抓住挤到一边的塌缩。
    """
    pts = list(corners)
    for i in range(4):
        if not _tri_plausible(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4], W, H):
            return False
    return True


def select_reliable_candidates(
    pts: list[Point], W: int, H: int
) -> list[Point]:
    """从 4 个有序角点里挑「构成合理三角形」的最大可靠锚点子集（2~3 点）。

    用于另一半把不构成近似四边形的 4 点重建为仿四边形：
      - 优先取能构成近似三角形（面积最大）的 3 点；
      - 没有则退化为取相距足够远的一对点（可作为近似对角线）；
      - 连一对足够远的点都找不到 → 返回空（调用方应视作塌缩兜底）。
    """
    from itertools import combinations

    diag = math.hypot(W, H)
    best_tri: list[Point] | None = None
    best_area = -1.0
    for i, j, k in combinations(range(4), 3):
        a, b, c = pts[i], pts[j], pts[k]
        if not _tri_plausible(a, b, c, W, H):
            continue
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        if area > best_area:
            best_area = area
            best_tri = [a, b, c]
    if best_tri is not None:
        return best_tri

    # 退化：没有合理三角形 → 取相距最远的一对点（近似对角线）
    best_pair: list[Point] | None = None
    best_d = -1.0
    for i, j in combinations(range(4), 2):
        d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
        if d > best_d:
            best_d = d
            best_pair = [pts[i], pts[j]]
    if best_pair is not None and best_d >= diag * MIN_ANCHOR_DIAG_FRACTION:
        return best_pair
    return []
