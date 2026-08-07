"""Moku RT-DETR 后处理 —— 移植自 TS `moku-postprocess.ts`。

包括：
  - preprocess: 图像 → (1,3,640,640) float32 张量（ImageNet 未归一化）
  - map_stones_to_grid: 虫洞检测中心通过单应映射到棋盘交叉点
  - postprocess: 解码 300 个查询 → 石头/角点 → 完整 RecognitionResult
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .corners import (
    are_corners_degenerate,
    inset_image_corners,
    order_corners,
    spread_collapsed_corners,
)
from .perspective import apply_homography, compute_homography, warp_perspective
from .sgf import build_sgf
from .types import (
    BoardCorners,
    DetectedStone,
    MokuRawDetection,
    Point,
    RecognitionResult,
)

INPUT_SIZE = 640
NUM_QUERIES = 300
NUM_CLASSES = 3

CLASS_BLACK_STONE = 0
CLASS_WHITE_STONE = 1
CLASS_BOARD_CORNER = 2

DEFAULT_THRESHOLD = 0.035
WARP_OUTPUT_SIZE = 800


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(img: "np.ndarray") -> np.ndarray:
    """将 HxWx4 RGBA 图 → (1,3,640,640) float32 CHW，像素范围 [0,1]。

    对齐 TS preprocess（do_normalize=false，仅除以 255）。
    """
    H, W = img.shape[0], img.shape[1]
    data = img.astype(np.float32) / 255.0
    rgb = data[..., :3]  # (H, W, 3)

    buf = np.zeros((3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    # 目标像素矩阵（双线性插值，映射回源坐标）
    ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) * (H / INPUT_SIZE) - 0.5
    xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) * (W / INPUT_SIZE) - 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    x0 = np.clip(np.floor(gx).astype(np.int32), 0, W - 1)
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (gx - np.floor(gx)).astype(np.float32)
    fy = (gy - np.floor(gy)).astype(np.float32)

    for c in range(3):
        v00 = rgb[y0, x0, c]
        v10 = rgb[y0, x1, c]
        v01 = rgb[y1, x0, c]
        v11 = rgb[y1, x1, c]
        val = (
            v00 * (1 - fx) * (1 - fy)
            + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy
            + v11 * fx * fy
        )
        buf[c] = val

    return buf[np.newaxis, ...]  # (1,3,640,640)


# ── 网格映射 ────────────────────────────────────────────────────────────────


def map_stones_to_grid(
    stones: list[MokuRawDetection],
    corners: BoardCorners,
    board_size: int,
) -> list[DetectedStone]:
    """将虫洞检测中心映射到离散交叉点。对齐 TS mapStonesToGrid。"""
    dst: tuple[Point, Point, Point, Point] = ((0, 0), (1, 0), (1, 1), (0, 1))
    Hmg = compute_homography(corners, dst)
    if Hmg is None:
        return []

    # 按 score 降序，高分优先占位
    sorted_stones = sorted(stones, key=lambda d: d.score, reverse=True)
    result: list[DetectedStone] = []
    occupied: set[tuple[int, int]] = set()

    for det in sorted_stones:
        rx, ry = apply_homography(Hmg, det.cx, det.cy)
        col = int(round(rx * (board_size - 1)))
        row = int(round(ry * (board_size - 1)))
        if col < 0 or col >= board_size or row < 0 or row >= board_size:
            continue
        if (col, row) in occupied:
            continue
        occupied.add((col, row))
        color = "black" if det.class_id == CLASS_BLACK_STONE else "white"
        result.append(DetectedStone(x=col, y=row, color=color))  # type: ignore[arg-type]

    return result


# ── 主后处理 ────────────────────────────────────────────────────────────────


def postprocess(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_img: "np.ndarray",
    board_size: int,
    threshold: float,
    output_size: int,
) -> RecognitionResult:
    """对齐 TS postprocess。logits/pred_boxes 为 (1,300,3)/(1,300,4) 展开后的。"""
    H, W = orig_img.shape[0], orig_img.shape[1]
    stones: list[MokuRawDetection] = []
    corner_candidates: list[MokuRawDetection] = []

    CORNER_MIN_THRESHOLD = 0.005

    # 展平到 (NUM_QUERIES, ...)
    logits = np.asarray(logits).reshape(NUM_QUERIES, NUM_CLASSES)
    pred_boxes = np.asarray(pred_boxes).reshape(NUM_QUERIES, 4)

    for q in range(NUM_QUERIES):
        logit = logits[q]
        sc = sigmoid(logit)
        best = int(np.argmax(sc))
        score = float(sc[best])
        min_score = CORNER_MIN_THRESHOLD if best == CLASS_BOARD_CORNER else threshold
        if score < min_score:
            continue
        cx = float(pred_boxes[q, 0]) * W
        cy = float(pred_boxes[q, 1]) * H
        det = MokuRawDetection(cx=cx, cy=cy, class_id=best, score=score)
        if best == CLASS_BOARD_CORNER:
            corner_candidates.append(det)
        else:
            stones.append(det)

    # 角点按置信度降序，取前 4
    corner_candidates.sort(key=lambda d: d.score, reverse=True)

    # 去重重叠角点（距离 < 5% 对角线 → 去低分）
    dedupe_min_dist = math.hypot(W, H) * 0.05
    deduped: list[MokuRawDetection] = []
    for det in corner_candidates:
        if not deduped or all(
            math.hypot(det.cx - o.cx, det.cy - o.cy) >= dedupe_min_dist for o in deduped
        ):
            deduped.append(det)

    if len(deduped) < 2:
        corners = inset_image_corners(W, H, 0.05)
        warped = warp_perspective(orig_img, corners, output_size)
        return RecognitionResult(
            board_size=board_size,
            stones=[],
            corners=corners,
            corners_detected=False,
            sgf=build_sgf(board_size, []),
            moku_raw_corners=None,
            moku_corner_count=len(deduped),
        )

    top4 = _infer_top4(deduped, W, H)

    # 排序为 TL→TR→BR→BL
    corners: BoardCorners = order_corners(top4)
    moku_raw_corners: BoardCorners = corners

    if are_corners_degenerate(corners, W, H):
        corners = inset_image_corners(W, H, 0.05)

    spread, _ = spread_collapsed_corners(corners, W, H)
    corners = spread

    # 透视校正：角点 → 内插 8% 边距的方形
    WARP_MARGIN = 0.08
    m = round(output_size * WARP_MARGIN)
    inset_dst: tuple[Point, Point, Point, Point] = (
        (m, m),
        (output_size - 1 - m, m),
        (output_size - 1 - m, output_size - 1 - m),
        (m, output_size - 1 - m),
    )
    warped = warp_perspective(orig_img, corners, output_size, inset_dst)

    estimated_grid = inset_dst
    detected_stones = map_stones_to_grid(stones, corners, board_size)

    return RecognitionResult(
        board_size=board_size,
        stones=detected_stones,
        corners=corners,
        corners_detected=True,
        sgf=build_sgf(board_size, detected_stones),
        estimated_grid_corners=estimated_grid,
        moku_raw_corners=moku_raw_corners,
        moku_corner_count=min(len(deduped), 4),
    )


def _infer_top4(
    corner_candidates: list[MokuRawDetection], W: int, H: int
) -> list[Point]:
    """根据 2/3/4 个候选角点推导出 4 个角点。对齐 TS。"""
    n = len(corner_candidates)
    pts = [(d.cx, d.cy) for d in corner_candidates]

    if n == 2:
        (p1x, p1y), (p2x, p2y) = pts
        mx = (p1x + p2x) / 2
        my = (p1y + p2y) / 2
        dx = p2x - p1x
        dy = p2y - p1y

        candidates: list[list[Point]] = []
        # 解释 1：对角（绕中心旋转 90°）
        hdx = dx / 2
        hdy = dy / 2
        candidates.append([p1 := (p1x, p1y), (mx + hdy, my - hdx), (p2x, p2y), (mx - hdy, my + hdx)])
        # 解释 2a：相邻 (+90°)
        candidates.append([(p1x, p1y), (p2x, p2y), (p2x - dy, p2y + dx), (p1x - dy, p1y + dx)])
        # 解释 2b：相邻 (-90°)
        candidates.append([(p1x, p1y), (p2x, p2y), (p2x + dy, p2y - dx), (p1x + dy, p1y - dx)])

        best_quad = candidates[0]
        best_score = float("-inf")
        for quad in candidates:
            inside = 0
            margin_sum = 0.0
            for (px, py) in quad:
                mxx = min(px, W - px)
                myy = min(py, H - py)
                if mxx >= 0 and myy >= 0:
                    inside += 1
                    margin_sum += mxx + myy
                else:
                    margin_sum += mxx + myy
            score = inside * 1e6 + margin_sum
            if score > best_score:
                best_score = score
                best_quad = quad
        return best_quad

    if n == 3:
        best_quad: list[Point] | None = None
        best_score = float("inf")
        for diag in range(3):
            a = pts[diag]
            b = pts[(diag + 1) % 3]
            c = pts[(diag + 2) % 3]
            p4: Point = (b[0] + c[0] - a[0], b[1] + c[1] - a[1])
            quad = [a, b, c, p4]
            d1 = math.hypot(a[0] - p4[0], a[1] - p4[1])
            d2 = math.hypot(b[0] - c[0], b[1] - c[1])
            score = abs(d1 - d2)
            if score < best_score:
                best_score = score
                best_quad = quad
        return best_quad  # type: ignore[return-value]

    return pts[:4]
