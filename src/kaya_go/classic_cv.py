"""经典 CV 角点检测 —— 移植自 TS `image.boardMask` / `corners.findBoardQuadrilateral`。

Moku RT-DETR 角点检测塌缩（如某一对角挤到图片一边）时，用这里基于饱和度掩码
的四边形极值点法重新识别棋盘四角，作为『用户手工角点』之前的可靠兜底。

只读依赖 numpy / Pillow；opencv/scipy 为可选增强（有则先用形态学+凸包精化，
否则走纯 numpy 的极值点法，二者结果都以 `orderCorners` 统一为 TL/TR/BR/BL）。
"""
from __future__ import annotations

import numpy as np

from .corners import order_corners
from .types import BoardCorners

# boardMask 孔径：饱和度/亮度阈值、形态学膨胀半径
SAT_THRESHOLD = 0.1
BRIGHT_MAX = 235
BRIGHT_MIN = 35
DILATE_RADIUS = 5

# 找到四边形后，精化时的多边形逼近参数（与 opencv approxPolyDP 的 epsilon 比例）
APPROX_EPS_FRACTION = 0.02

# 掩码面积占图像面积的下限：不足则判定未检出棋盘
MIN_AREA_FRACTION = 0.05


def _to_gray(rgba: np.ndarray) -> np.ndarray:
    """RGBA (H,W,4) uint8 → (H,W) float32 灰度，对齐 TS toGrayscale。"""
    data = rgba.astype(np.float32)
    return 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]


def _saturation(rgba: np.ndarray) -> np.ndarray:
    """RGBA → (H,W) float32 饱和度（max/min，对齐 TS computeSaturation）。"""
    rgb = rgba[..., :3].astype(np.float32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    eps = np.finfo(np.float32).eps
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, eps), 0)


def _board_mask(rgba: np.ndarray) -> np.ndarray:
    """棋盘调色像素掩码，并膨胀填洞（对齐 TS boardMask）。"""
    gray = _to_gray(rgba)
    sat = _saturation(rgba)
    mask = (
        (sat > SAT_THRESHOLD)
        & (gray < BRIGHT_MAX)
        & (gray > BRIGHT_MIN)
    ).astype(np.uint8)

    # 形态学膨胀，填充石头空洞（石头本身不饱和但位于棋盘上）
    kernel = np.ones((2 * DILATE_RADIUS + 1, 2 * DILATE_RADIUS + 1), np.uint8)
    try:
        import cv2

        return cv2.dilate(mask, kernel, iterations=1)
    except ImportError:
        # 纯 numpy 膨胀（无 opencv 时）：逐偏移或移位累加
        k = DILATE_RADIUS
        H, W = mask.shape
        dilated = np.zeros_like(mask)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                shifted = np.zeros_like(mask)
                src_y, src_x = max(0, -dy), max(0, -dx)
                dst_y, dst_x = max(0, dy), max(0, dx)
                hh = H - abs(dy)
                ww = W - abs(dx)
                shifted[dst_y:dst_y + hh, dst_x:dst_x + ww] = mask[
                    src_y:src_y + hh, src_x:src_x + ww
                ]
                dilated |= shifted
        return dilated


def _find_board_quadrilateral(mask: np.ndarray) -> BoardCorners | None:
    """由棋盘掩码边界像素的极值点推断四角，对齐 TS findBoardQuadrilateral。

    返回已按 TL/TR/BR/BL 排序的四角（图像坐标）；面积不足/未检出返回 None。
    """
    H, W = mask.shape
    # 边界像素 = 掩码为 1 但四邻不全为 1
    padded = np.pad(mask, 1)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = (mask == 1) & ~interior
    ys, xs = np.nonzero(boundary)
    boundary_count = len(xs)
    if boundary_count < 20:
        return None

    tl = _extreme(xs, ys, "tl")
    br = _extreme(xs, ys, "br")
    tr = _extreme(xs, ys, "tr")
    bl = _extreme(xs, ys, "bl")

    # 四边形必须覆盖图像 ≥5% 面积，否则视为未检出
    area = (
        abs((tr[0] - tl[0]) * (br[1] - tl[1]) - (br[0] - tl[0]) * (tr[1] - tl[1])) / 2
        + abs((br[0] - tl[0]) * (bl[1] - tl[1]) - (bl[0] - tl[0]) * (br[1] - tl[1])) / 2
    )
    if area < W * H * MIN_AREA_FRACTION:
        return None

    return order_corners([tl, tr, br, bl])


def _extreme(xs: np.ndarray, ys: np.ndarray, corner: str) -> tuple[int, int]:
    """棋盘掩码边界像素里取某个几何最值角点 (x, y)。

    xs/ys 来自同一组互相配对的索引（np.nonzero），因此 argmin/argmax 的位置
    天然对齐——直接用各项全局最值下标取另一数组的值即可，等价于 TS reduce 的
    Object.assign 覆盖写法，没有坐标错配风险。

    规则与 Kaya `findBoardQuadrilateral` 一致：
      - TL：min(x + y)；BR：max(x + y)
      - TR：max(x − y)；BL：min(x − y)
    """
    if corner == "tl":
        return int(xs[(xs + ys).argmin()]), int(ys[(xs + ys).argmin()])
    if corner == "br":
        return int(xs[(xs + ys).argmax()]), int(ys[(xs + ys).argmax()])
    if corner == "tr":
        d = (xs - ys).argmax()
        return int(xs[d]), int(ys[d])
    d = (xs - ys).argmin()
    return int(xs[d]), int(ys[d])


def _refine_corners_with_cv(rgba: np.ndarray, mask: np.ndarray) -> BoardCorners | None:
    """用 opencv 对棋盘掩码做轮廓 → 凸包 → 多边形逼近，得到更贴合畸变的四角。

    仅在 opencv 可用时调用；逼近失败或 n≠4 时返回 None，由调用方退回到
    极值点法结果。
    """
    try:
        import cv2
    except ImportError:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < mask.shape[0] * mask.shape[1] * MIN_AREA_FRACTION:
        return None
    hull = cv2.convexHull(big)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    quad = cv2.approxPolyDP(hull, APPROX_EPS_FRACTION * peri, True)
    if len(quad) != 4:
        return None
    pts = [(int(p[0]), int(p[1])) for p in quad.reshape(-1, 2)]
    return order_corners(pts)


def _quad_mask(h: int, w: int, quad: BoardCorners) -> np.ndarray:
    """在 (h,w) 画布上把 4 角多边形填充为白(1)，外部置黑(0)。

    供 `find_board_corners(..., mask_corners=...)` 把经典 CV 的搜索范围
    约束在 moku 的仿四边形内——CV 检出的轮廓必然落在该四边形内，不会被
    整图或棋盘外的色块干扰。
    """
    pts = np.array([quad], dtype=np.float32)
    mask = np.zeros((h, w), np.uint8)
    try:
        import cv2

        cv2.fillPoly(mask, np.round(pts).astype(np.int32), 1)
    except ImportError:
        # 纯 numpy 扫描线填充：对每行，把落在四边形内的 x 区间置 1
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        inside = _points_inside_quad(pts[0], xs, ys)
        mask[inside] = 1
    return mask


def _points_inside_quad(quad, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """射线法：判断网格点是否落在凸四边形内。quad: (4,2) float。"""
    px, py = quad[:, 0], quad[:, 1]
    crossings = np.zeros(xs.shape, dtype=np.int32)
    n = 4
    for i in range(n):
        x1, y1 = px[i], py[i]
        x2, y2 = px[(i + 1) % n], py[(i + 1) % n]
        # 边 (p1→p2) 跨过水平射线：仅当一端在网格点上方、另一端下方
        cond = (ys > y1) != (ys > y2)
        # 交点 x 坐标
        x_isect = (x2 - x1) * (ys - y1) / ((y2 - y1) + 1e-12) + x1
        crossings += cond & (xs < x_isect)
    return crossings % 2 == 1


def find_board_corners(
    rgba: np.ndarray, mask_corners: BoardCorners | None = None
) -> BoardCorners | None:
    """在输入的 RGBA 原图（H,W,4 uint8）上检测棋盘四角。

    优先 opencv 精化（更贴合透视畸变）；否则用纯 numpy 极值点法。
    检出即返回 TL/TR/BR/BL，未检出返回 None。

    `mask_corners`：可选的搜索范围约束（TL/TR/BR/BL，图像坐标）。传了就把
    饱和度掩码限定在沿该四边形填充的多边形内部——用于 Moku 重建出仿四边形后，
    让 CV 只在那个可信区域内重识别。
    """
    H, W = rgba.shape[0], rgba.shape[1]
    # 缩小到与 Kaya 一致的 600px 侧做检测（大图噪声多、开销高）
    target = 600
    if max(H, W) > target:
        scale = target / max(H, W)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        try:
            import cv2

            small = cv2.resize(rgba[..., :3], (nw, nh), interpolation=cv2.INTER_AREA)
            small_rgba = np.dstack(
                [small, np.full((nh, nw, 1), 255, np.uint8)]
            )
        except ImportError:
            small_rgba = _resize_nearest(rgba, nw, nh)
    else:
        small_rgba = rgba
        scale = 1.0

    mask = _board_mask(small_rgba)

    # 可选：把搜索范围约束到 moku 仿四边形内（mask 需与缩放同步到检测坐标系）。
    # 目标是让经典 CV 只在仿四边形范围内找轮廓，不被棋盘外色块干扰。
    if mask_corners is not None:
        if scale != 1.0:
            scaled_mask = tuple(
                (x * scale, y * scale) for x, y in mask_corners
            )  # type: ignore[assignment]
            scope = _quad_mask(nh, nw, scaled_mask)
        else:
            scope = _quad_mask(H, W, mask_corners)
        mask = mask & scope
    corners = _refine_corners_with_cv(small_rgba, mask)
    if corners is None:
        corners = _find_board_quadrilateral(mask)

    if corners is None:
        return None

    # 缩放回原图坐标
    if scale != 1.0:
        inv = 1.0 / scale
        corners = tuple(
            (int(round(x * inv)), int(round(y * inv))) for x, y in corners
        )  # type: ignore[assignment]
    return corners  # type: ignore[return-value]


def _resize_nearest(rgba: np.ndarray, nw: int, nh: int) -> np.ndarray:
    """无 opencv 时用 numpy 最近邻缩放（只用于掩码检测的降采样，精度要求低）。"""
    H, W = rgba.shape[0], rgba.shape[1]
    ys = np.floor(np.linspace(0, H - 1, nh)).astype(np.int32)
    xs = np.floor(np.linspace(0, W - 1, nw)).astype(np.int32)
    return rgba[np.ix_(ys, xs)]
