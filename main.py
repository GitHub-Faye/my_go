"""Kaya 围棋棋盘识别 —— FastAPI 接口。

把 @kaya/board-recognition（浏览器端）的 "识别图片 → 棋盘 SGF" 能力移植为
HTTP 服务。核心推理走 ONNX Runtime + moku-v3 RT-DETR 模型。

接口约定与浏览器端 MokuDetector.detect 一致：
  - 输入  multipart 图片 + 可选 boardSize / threshold
  - 输出  RecognitionResult（stones / corners / sgf 等）
"""
from __future__ import annotations

import base64
import logging
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from kaya_go.deadstones import (
    count_dead_stones,
    derive_dead_stones,
    get_probability_map,
)
from kaya_go.detector import MokuDetector
from kaya_go.scoring import (
    compute_estimated_territory,
    compute_territory,
    parse_dead_stones,
)
from kaya_go.moku_postprocess import DEFAULT_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kaya Go Board Recognition",
    description="基于 ONNX RT-DETR (moku-v3) 的围棋棋盘照片识别，返回棋盘状态与 SGF。",
    version="0.1.0",
    openapi_url="/api/openapi.json",
    docs_url="/docs",
)

# 模型路径：优先环境变量 KAYA_MOKU_MODEL，否则 my_go/models/moku-v3.onnx
MODEL_PATH = Path(os.environ.get("KAYA_MOKU_MODEL") or "models/moku-v3.onnx")

# 全局单例会话（线程安全；模型加载一次）
_detector: MokuDetector | None = None

# 死子估计迭代次数默认值 —— 与前端 INITIAL_ITERATIONS=2000 一致
DEFAULT_ITERATIONS = 2000

# 合法棋盘尺寸:9 / 13 / 19 路
VALID_SIZES = (9, 13, 19)


def _validate_sign_map(rows: list[list[int]]) -> tuple[int, int]:
    """校验 signMap 为 9/13/19 的正方形矩阵、取值仅 -1/0/1。

    先做防御性形状校验:inhomogeneous 的 list 会在 np.asarray 阶段抛 ValueError,
    需在构造 numpy 数组前拦截,统一返回 400。返回 (height, width)。
    """
    if not rows or len(rows) not in VALID_SIZES or any(
        not isinstance(r, list) or len(r) != len(rows) for r in rows
    ):
        raise HTTPException(status_code=400, detail="signMap 须为 9/13/19 的正方形矩阵")
    board = np.asarray(rows, dtype=np.int8)
    h, w = board.shape
    vals = set(np.unique(board).tolist())
    if not vals.issubset({-1, 0, 1}):
        raise HTTPException(status_code=400, detail="signMap 只允许取值 1/-1/0")
    return h, w


class DeadStonesRequest(BaseModel):
    """死子估计请求体。signMap 为 boardSize×boardSize 矩阵 1=黑/-1=白/0=空。

    可直接传入 `/api/v1/recognize` 响应里的 `signMap`。
    """

    signMap: list[list[int]] = Field(..., description="棋盘状态矩阵 signMap[y][x]")
    iterations: int = Field(default=DEFAULT_ITERATIONS, ge=1, le=100000)
    seed: int | None = Field(default=None, description="随机种子；空则取当前时间")


class DeadStonesResponse(BaseModel):
    boardSize: int
    probabilityMap: list[list[float]] | None = None
    deadStones: list[dict[str, int]]
    blackDeadStones: int
    whiteDeadStones: int


class ScoreRequest(BaseModel):
    """记分请求体。deadStones 来自 `/api/v1/deadstones` 响应;komi 为白方贴目。

    可传 `probabilityMap`(来自 deadstones 响应)走估计路径;不传则只用
    flood fill 精确路径——此时 `useEstimated` 被忽略。
    """

    signMap: list[list[int]] = Field(..., description="棋盘状态矩阵 signMap[y][x]")
    deadStones: list[dict[str, int]] = Field(
        default_factory=list, description="死子坐标列表 [{x, y}, ...]"
    )
    komi: float = Field(default=0.0, description="白方贴目")
    probabilityMap: list[list[float]] | None = Field(
        default=None, description="领地概率图(可选;传则走估计路径)"
    )
    useEstimated: bool = Field(
        default=True, description="是否用 probabilityMap 补单官(默认开,忽略单官为中立)"
    )


class ScoreResponse(BaseModel):
    boardSize: int
    territoryMap: list[list[int]]
    blackTerritory: int
    whiteTerritory: int
    blackCaptures: int = 0
    whiteCaptures: int = 0
    whiteDeadStones: int
    blackDeadStones: int
    komi: float
    blackScore: float
    whiteScore: float


def get_detector() -> MokuDetector:
    """懒加载单例。模型缺失时给出可读错误。"""
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"模型文件不存在：{MODEL_PATH}（请将 moku-v3.onnx 放入 models/ 或设置 KAYA_MOKU_MODEL）",
            )
        _detector = MokuDetector(MODEL_PATH)
        logger.info("Moku 模型已加载：%s", MODEL_PATH)
    return _detector


@app.get("/")
def read_root():
    return {"message": "Hello, Kaya!"}


@app.get("/health")
def health():
    model_ok = MODEL_PATH.exists()
    return {
        "status": "ok" if model_ok else "degraded",
        "model_loaded": (model_ok and _detector is not None),
        "model_exists": model_ok,
    }


@app.post("/api/v1/corners")
async def corners_only(
    image: UploadFile = File(...),
):
    """用 Moku RT-DETR 识别棋盘 4 角（独立于棋盘状态识别流程）。

    流程（对齐用户设计）：
      ① Moku 检角点类候选 → 去重。去重后不足 4 点 → 直接 400 报错
        （而不是用 2/3 点硬推，避免造出不可靠的角）。
      ② 取 top4 按 TL → TR → BR → BL 排序。若 4 点构成近似四边形
        （任取 3 点构成的三角形都非退化），直接返回，cornersDetected=true。
      ③ 若 4 点不构成近似四边形：用其中可靠的锚点子集（优先 3 点、退而
        求其次 2 点），求中心点并用镜像/平行四边形中点公式重建缺失点，
        得到仿四边形。
      ④ 把仿四边形丢给经典 CV 二次矫正（轮廓凸包多边形逼近，更贴合畸变）。
        CV 检得出 → 用 CV 结果；CV 检不出 → 返回仿四边形本身。
        （不退回整盘内缩——那会丢掉可靠的锚点。）

    响应：
      - `corners`      [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，原图坐标
      - `order`        固定 "TL/TR/BR/BL"
      - `cornersDetected`  直接用 moku 4 角（true）/ 重建+CV 矫正（false）
      - `mokuRawCorners`   Moku 推断出的原始四角（供前端对照），重建时非空

    复用 moku_postprocess 的角点推断逻辑，保证两端输出一致。
    """
    detector = get_detector()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")

    try:
        pil = Image.open(BytesIO(raw))
        pil.load()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析图片：{e}")

    # 统一为 RGBA uint8 (H, W, 4) —— 与 /api/v1/recognize 一致
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    arr = np.asarray(pil)

    # 跑一次推理，直接取角点类的检测，用与 recognize 相同的推断逻辑
    logits, pred_boxes = detector._run_inference(arr)
    import math

    from kaya_go.classic_cv import find_board_corners
    from kaya_go.corners import (
        order_corners,
        quad_is_plausible,
        select_reliable_candidates,
    )
    from kaya_go.moku_postprocess import (
        _infer_top4,
        CLASS_BOARD_CORNER,
        NUM_QUERIES,
        sigmoid,
    )
    from kaya_go.types import MokuRawDetection

    H, W = arr.shape[0], arr.shape[1]
    logits = np.asarray(logits).reshape(NUM_QUERIES, 3)
    pred_boxes = np.asarray(pred_boxes).reshape(NUM_QUERIES, 4)

    corner_candidates: list[MokuRawDetection] = []
    for q in range(NUM_QUERIES):
        sc = sigmoid(logits[q])
        best = int(np.argmax(sc))
        score = float(sc[best])
        if best == CLASS_BOARD_CORNER and score >= 0.005:
            cx = float(pred_boxes[q, 0]) * W
            cy = float(pred_boxes[q, 1]) * H
            corner_candidates.append(
                MokuRawDetection(cx=cx, cy=cy, class_id=best, score=score)
            )

    # 去重（5% 对角线内的重叠角点取高分）
    corner_candidates.sort(key=lambda d: d.score, reverse=True)
    dedupe_min_dist = math.hypot(W, H) * 0.05
    deduped: list[MokuRawDetection] = []
    for det in corner_candidates:
        if not deduped or all(
            math.hypot(det.cx - o.cx, det.cy - o.cy) >= dedupe_min_dist
            for o in deduped
        ):
            deduped.append(det)

    # ① 不足 4 点 → 直接报错（不硬推，避免造出不可靠角点）
    if len(deduped) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Moku 仅检出 {len(deduped)} 个角点（不足 4 个），无法识别棋盘。"
            "请换更清晰的棋盘照片，或手动拖 4 角。",
        )

    top4 = list(_infer_top4(deduped, W, H))
    corners = order_corners(top4)
    moku_raw_corners = [list(pt) for pt in corners]

    # ② 4 点构成近似四边形 → 直接用
    if quad_is_plausible(corners, W, H):
        return {
            "corners": [[x, y] for x, y in corners],
            "order": "TL/TR/BR/BL",
            "cornersDetected": True,
            "mokuRawCorners": moku_raw_corners,
        }

    # ③ 4 点不构成近似四边形：挑可靠锚点（3 点优先、2 点兜底）重建仿四边形
    reliable = select_reliable_candidates(list(corners), W, H)
    if len(reliable) >= 2:
        reconstructed = _infer_top4(
            [
                MokuRawDetection(cx=x, cy=y, class_id=CLASS_BOARD_CORNER, score=0.0)
                for x, y in reliable
            ],
            W,
            H,
        )
        recovered = order_corners(reconstructed)
        corners = recovered
    else:
        raise HTTPException(
            status_code=400,
            detail="Moku 检出的角点过于塌缩，找不到可靠锚点重建棋盘，请手动拖 4 角。",
        )

    # ④ 仿四边形作为搜索范围(mask)约束，丢给经典 CV 在范围内二次矫正；
    #    CV 在 mask 内检得出 → 用 CV 结果；检不出 → 以仿四边形为准（不掉
    #    moku 可靠锚点，也不整盘内缩）。
    cv_corners = find_board_corners(arr, mask_corners=recovered)
    if cv_corners is not None:
        corners = cv_corners
    else:
        corners = recovered
    return {
        "corners": [[x, y] for x, y in corners],
        "order": "TL/TR/BR/BL",
        "cornersDetected": False,
        "mokuRawCorners": moku_raw_corners,
        "rebuilt": True,
    }


@app.post("/api/v1/recognize")
async def recognize(
    image: UploadFile = File(...),
    boardSize: int = Query(default=19, ge=9, le=19),
    threshold: float = Query(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0),
    corners: str | None = Query(default=None, description="可选手工角点：x1,y1,x2,y2,x3,y3,x4,y4（TL/TR/BR/BL 顺时针）"),
):
    """识别一张围棋棋盘照片，返回棋盘状态、signMap、角点坐标与 SGF。

    - `image`  PNG/JPEG/WebP 等 Pillow 支持的格式
    - `boardSize` 9 / 13 / 19（默认 19）
    - `threshold` 石头检出置信度（默认 0.035）
    - `corners` 可选；前端拖好 4 角后传入，服务端用用户角点替代 Moku 自动角点

    响应在原有结果上新增三块供前端本地微调：
      - `detections`  过阈值检出的原始 (class/score/cx/cy) —— 本地拖阈值 re-filter
      - `warpedGray`  warped 灰度小图 base64 —— 本地重采样/标记黑白空
      - `gridCorners` warped 坐标内的网格四角

    返回的 `signMap` 为 boardSize×boardSize 矩阵：1=黑、-1=白、0=空
    （signMap[y][x]），可直接作为死子估计接口的输入。
    """
    detector = get_detector()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")

    try:
        pil = Image.open(BytesIO(raw))
        pil.load()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析图片：{e}")

    # 统一为 RGBA uint8 (H, W, 4)
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    arr = np.asarray(pil)  # (H, W, 4)

    # 解析用户角点（可选）：x1,y1,x2,y2,x3,y3,x4,y4 → [[x1,y1],...,[x4,y4]]
    user_corners = None
    if corners:
        parts = [float(p) for p in corners.split(",")]
        if len(parts) != 8:
            raise HTTPException(status_code=400, detail="corners 须为 x1,y1,x2,y2,x3,y3,x4,y4")
        user_corners = (
            (parts[0], parts[1]),
            (parts[2], parts[3]),
            (parts[4], parts[5]),
            (parts[6], parts[7]),
        )

    try:
        result, raw_detections, warped_gray = detector.detect_full(
            arr,
            board_size=boardSize,
            threshold=threshold,
            corners=user_corners,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("识别失败")
        raise HTTPException(status_code=500, detail=f"识别失败：{e}")

    body = result.to_dict()
    # 新增中间产物（前后端分离：本地 refilter / 标记黑白空）
    body["detections"] = [
        {"class": d.class_id, "score": round(d.score, 4), "cx": d.cx, "cy": d.cy}
        for d in raw_detections
    ]
    # warped 灰度小图 → 单通道 uint8 PNG base64
    from PIL import Image as PILImage

    gray_img = PILImage.fromarray(warped_gray, mode="L")
    buf = BytesIO()
    gray_img.save(buf, format="PNG")
    body["warpedGray"] = {
        "width": gray_img.width,
        "height": gray_img.height,
        "dataBase64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }
    if result.estimated_grid_corners:
        body["gridCorners"] = [list(pts) for pts in result.estimated_grid_corners]
    else:
        body["gridCorners"] = None

    return body


@app.post("/api/v1/deadstones")
def deadstones(req: DeadStonesRequest):
    """由棋盘状态(signMap)估计死子 → 纯 Monte Carlo,不依赖 ONNX 模型。

    - 输入  `signMap`(来自 /api/v1/recognize 的响应) + 可选 iterations / seed
    - 输出  `probabilityMap`(领地概率图 float ∈ [-1,1],正=黑控制)+ `deadStones`
        + 各色死子计数。作为记分的前置:同 `probabilityMap`(原始盘)命中死子后,
        记分端应先在死子清零的盘上重跑本接口,再取 `probabilityMap` 走估计路径。

    iterations 语义与前端一致:前一半局白先手、后一半黑先手。
    """
    h, w = _validate_sign_map(req.signMap)
    board = np.asarray(req.signMap, dtype=np.int8)

    seed = req.seed if req.seed is not None else int(time.time() * 1000) % 0xFFFFFFFF
    prob = get_probability_map(board, req.iterations, seed)

    dead = derive_dead_stones(prob, req.signMap)
    counts = count_dead_stones(req.signMap, dead)

    return DeadStonesResponse(
        boardSize=h,
        probabilityMap=prob.tolist(),
        deadStones=dead,
        **counts,
    )


@app.post("/api/v1/score")
def score(req: ScoreRequest):
    """由 signMap + deadStones(可带 probabilityMap)计算领地与分数。

    - `signMap`/`deadStones`/`probabilityMap` 均来自 `/api/v1/deadstones` 的响应链,
      其中 `probabilityMap` 需与移除死子后的局面匹配才准确(即命中死子后,先在死子
      清零的盘上重跑 deadstones 接口)。
    - `komi` 计入白方得分(黑方先行的贴目。按 Kaya 前端 ScoreEstimator 约定)。
    - 记分语义对应 Kaya 前端 scoring.ts:死子清零后 flood fill 判封口领地,
      剩余单官点若提供概率图再用 ±0.2 阈值补齐(估计路径)。

    响应含 territoryMap(1=黑地 / -1=白地 / 0=单官)与黑白最终得分。
    """
    h, w = _validate_sign_map(req.signMap)

    dead = parse_dead_stones(req.deadStones)
    for x, y in dead:
        if not (0 <= x < w and 0 <= y < h):
            raise HTTPException(status_code=400, detail=f"死子坐标越界:({x},{y})")

    # 死子计数 + 领地计算统一委托领域模块,避免与 /deadstones 端点各算一套
    counts = count_dead_stones(req.signMap, req.deadStones)

    if req.useEstimated and req.probabilityMap is not None:
        result = compute_estimated_territory(req.signMap, req.probabilityMap, dead)
    else:
        result = compute_territory(req.signMap, dead)

    black_territory = result["blackTerritory"]
    white_territory = result["whiteTerritory"]
    black_dead = counts["blackDeadStones"]
    white_dead = counts["whiteDeadStones"]

    # 分数,对齐前端 ScoreEstimator:黑方得「白死子 + 黑地」,白方得「黑死子 + 白地 + komi」。
    # capture(提子数)由对局历史提供,HTTP 场景未知,置 0。
    black_score = black_territory + white_dead
    white_score = white_territory + black_dead + req.komi

    return ScoreResponse(
        boardSize=h,
        territoryMap=result["territories"],
        blackTerritory=black_territory,
        whiteTerritory=white_territory,
        blackCaptures=0,
        whiteCaptures=0,
        blackDeadStones=black_dead,
        whiteDeadStones=white_dead,
        komi=req.komi,
        blackScore=black_score,
        whiteScore=white_score,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
