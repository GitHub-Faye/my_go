"""透视变换与单应矩阵 —— 移植自 TS `perspective.ts`。

compute_homography / apply_homography / invert_matrix3 / warp_perspective。
"""
from __future__ import annotations

from .types import BoardCorners, Point

EPS = 1e-12


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float] | None:
    """高斯消元求解 8x8 线性系统，奇异返回 None。对齐 TS solveLinear。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # 部分主元
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < EPS:
            return None
        for row in range(n):
            if row == col:
                continue
            f = M[row][col] / M[col][col]
            for k in range(col, n + 1):
                M[row][k] -= f * M[col][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def compute_homography(
    src: tuple[Point, Point, Point, Point], dst: tuple[Point, Point, Point, Point]
) -> list[float] | None:
    """计算 3x3 单应矩阵（9 元素，行主序，h8=1）。对齐 TS computeHomography。"""
    A: list[list[float]] = []
    b: list[float] = []
    for i in range(4):
        sx, sy = src[i]
        dx, dy = dst[i]
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        b.append(dx)
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
        b.append(dy)
    h = _solve_linear(A, b)
    if h is None:
        return None
    return h + [1.0]


def apply_homography(H: list[float], x: float, y: float) -> Point:
    """将点 (x,y) 经 H 映射 → (x',y')。"""
    w = H[6] * x + H[7] * y + H[8]
    if abs(w) < EPS:
        return (x, y)
    return ((H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w)


def invert_matrix3(m: list[float]) -> list[float] | None:
    """3x3 矩阵求逆。对齐 TS invertMatrix3。"""
    det = (
        m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
    )
    if abs(det) < EPS:
        return None
    inv = 1.0 / det
    return [
        (m[4] * m[8] - m[5] * m[7]) * inv,
        (m[2] * m[7] - m[1] * m[8]) * inv,
        (m[1] * m[5] - m[2] * m[4]) * inv,
        (m[5] * m[6] - m[3] * m[8]) * inv,
        (m[0] * m[8] - m[2] * m[6]) * inv,
        (m[2] * m[3] - m[0] * m[5]) * inv,
        (m[3] * m[7] - m[4] * m[6]) * inv,
        (m[1] * m[6] - m[0] * m[7]) * inv,
        (m[0] * m[4] - m[1] * m[3]) * inv,
    ]


def warp_perspective(
    rgba: "object",
    corners: BoardCorners,
    out_size: int,
    dst_corners: tuple[Point, Point, Point, Point] | None = None,
) -> "object":
    """逆映射双线性插值透视校正。返回 numpy RGBA 图。

    rgba: numpy.ndarray，shape (height, width, 4)，uint8。
    """
    import numpy as np  # 延迟导入，仅当真正需要透视时

    H, W = rgba.shape[0], rgba.shape[1]
    if dst_corners is None:
        dst: tuple[Point, Point, Point, Point] = (
            (0, 0),
            (out_size - 1, 0),
            (out_size - 1, out_size - 1),
            (0, out_size - 1),
        )
    else:
        dst = dst_corners

    Hmt = compute_homography(corners, dst)
    Hin = invert_matrix3(Hmt) if Hmt else None

    y_idx, x_idx = np.meshgrid(np.arange(out_size), np.arange(out_size), indexing="ij")
    out = np.zeros((out_size, out_size, 4), dtype=np.uint8)

    if Hin is None:
        # 退化为直接缩放
        sx = (x_idx.astype(float) / (out_size - 1)) * (W - 1)
        sy = (y_idx.astype(float) / (out_size - 1)) * (H - 1)
    else:
        import math

        w_ = Hin[6] * x_idx + Hin[7] * y_idx + Hin[8]
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = np.where(
                np.abs(w_) < EPS, x_idx, (Hin[0] * x_idx + Hin[1] * y_idx + Hin[2]) / w_
            )
            sy = np.where(
                np.abs(w_) < EPS, y_idx, (Hin[3] * x_idx + Hin[4] * y_idx + Hin[5]) / w_
            )

    sx = sx.astype(np.float64)
    sy = sy.astype(np.float64)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    dx = sx - x0
    dy = sy - y0
    x1 = x0 + 1
    y1 = y0 + 1

    # 越界采样 → 最近邻 + clamp
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)
    cx = np.clip(np.rint(sx).astype(np.int32), 0, W - 1)
    cy = np.clip(np.rint(sy).astype(np.int32), 0, H - 1)
    out[..., 3] = 255

    for c in range(3):
        nn = rgba[cy, cx, c].astype(np.float64)
        i00 = rgba[np.clip(y0, 0, H - 1), np.clip(x0, 0, W - 1), c].astype(np.float64)
        i10 = rgba[np.clip(y0, 0, H - 1), np.clip(x1, 0, W - 1), c].astype(np.float64)
        i01 = rgba[np.clip(y1, 0, H - 1), np.clip(x0, 0, W - 1), c].astype(np.float64)
        i11 = rgba[np.clip(y1, 0, H - 1), np.clip(x1, 0, W - 1), c].astype(np.float64)
        bilinear = (
            i00 * (1 - dx) * (1 - dy)
            + i10 * dx * (1 - dy)
            + i01 * (1 - dx) * dy
            + i11 * dx * dy
        )
        out[..., c] = np.where(valid, np.clip(bilinear, 0, 255), nn).astype(np.uint8)

    return out
