"""Kaya 围棋棋盘识别 —— FastAPI 接口。

把 @kaya/board-recognition（浏览器端）的 "识别图片 → 棋盘 SGF" 能力移植为
HTTP 服务。核心推理走 ONNX Runtime + moku-v3 RT-DETR 模型。

接口约定与浏览器端 MokuDetector.detect 一致：
  - 输入  multipart 图片 + 可选 boardSize / threshold
  - 输出  RecognitionResult（stones / corners / sgf 等）
"""
from __future__ import annotations

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
    """记分请求体。deadStones 来自 `/api/v1/deadstones` 响应;komi 为黑方贴目。

    可传 `probabilityMap`(来自 deadstones 响应)走估计路径;不传则只用
    flood fill 精确路径——此时 `useEstimated` 被忽略。
    """

    signMap: list[list[int]] = Field(..., description="棋盘状态矩阵 signMap[y][x]")
    deadStones: list[dict[str, int]] = Field(
        default_factory=list, description="死子坐标列表 [{x, y}, ...]"
    )
    komi: float = Field(default=0.0, description="黑方贴目")
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


@app.post("/api/v1/recognize")
async def recognize(
    image: UploadFile = File(...),
    boardSize: int = Query(default=19, ge=9, le=19),
    threshold: float = Query(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0),
):
    """识别一张围棋棋盘照片，返回棋盘状态、signMap、角点坐标与 SGF。

    - `image`  PNG/JPEG/WebP 等 Pillow 支持的格式
    - `boardSize` 9 / 13 / 19（默认 19）
    - `threshold` 石头检出置信度（默认 0.035）

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

    try:
        result = detector.detect(arr, board_size=boardSize, threshold=threshold)
    except Exception as e:  # noqa: BLE001
        logger.exception("识别失败")
        raise HTTPException(status_code=500, detail=f"识别失败：{e}")

    return result.to_dict()


@app.post("/api/v1/deadstones")
def deadstones(req: DeadStonesRequest):
    """由棋盘状态(signMap)估计死子 → 纯 Monte Carlo,不依赖 ONNX 模型。

    - 输入  `signMap`(来自 /api/v1/recognize 的响应) + 可选 iterations / seed
    - 输出  `probabilityMap`(领地概率图 float ∈ [-1,1],正=黑控制)+ `deadStones`
        + 各色死子计数,另含 `boardWithoutDead`(移除死子后的局面,供二次迭代)

    iterations 语义与前端一致:前一半局白先手、后一半黑先手。
    """
    rows = req.signMap
    # 先做防御性形状校验(inhomogeneous 的 list 会在 np.asarray 阶段抛 ValueError,
    # 需在构造 numpy 数组前拦截,统一返回 400)。
    if not rows or len(rows) not in (9, 13, 19) or any(
        not isinstance(r, list) or len(r) != len(rows) for r in rows
    ):
        raise HTTPException(status_code=400, detail="signMap 须为 9/13/19 的正方形矩阵")
    board = np.asarray(rows, dtype=np.int8)
    h, w = board.shape
    vals = set(np.unique(board).tolist())
    if not vals.issubset({-1, 0, 1}):
        raise HTTPException(status_code=400, detail="signMap 只允许取值 1/-1/0")

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
      其中 `probabilityMap` 需与移除死子后的局面匹配才准确。
    - `komi` 计入黑方得分(白方获胜时的贴目退让)。
    - 记分语义对应 Kaya 前端 scoring.ts:死子清零后 flood fill 判封口领地,
      剩余单官点若提供概率图再用 ±0.2 阈值补齐(估计路径)。

    响应含 territoryMap(1=黑地 / -1=白地 / 0=单官)与黑白最终得分。
    """
    rows = req.signMap
    if not rows or len(rows) not in (9, 13, 19) or any(
        not isinstance(r, list) or len(r) != len(rows) for r in rows
    ):
        raise HTTPException(status_code=400, detail="signMap 须为 9/13/19 的正方形矩阵")
    board = np.asarray(rows, dtype=np.int8)
    h, w = board.shape
    vals = set(np.unique(board).tolist())
    if not vals.issubset({-1, 0, 1}):
        raise HTTPException(status_code=400, detail="signMap 只允许取值 1/-1/0")

    dead = parse_dead_stones(req.deadStones)
    # 与死子接口输出一致,再核对一次取值范围
    for x, y in dead:
        if not (0 <= x < w and 0 <= y < h):
            raise HTTPException(status_code=400, detail=f"死子坐标越界:({x},{y})")

    # deadStones 计数:与 /api/v1/deadstones 的计数保持一致(按子色统计)
    black_dead = sum(1 for x, y in dead if rows[y][x] == 1)
    white_dead = sum(1 for x, y in dead if rows[y][x] == -1)

    if req.useEstimated and req.probabilityMap is not None:
        result = compute_estimated_territory(req.signMap, req.probabilityMap, dead)
    else:
        result = compute_territory(req.signMap, dead)

    black_territory = result["blackTerritory"]
    white_territory = result["whiteTerritory"]

    # 分数:领地 + 贴目(黑方得贴目)。capture 由对局历史提供,HTTP 场景未知,置 0。
    black_score = black_territory + black_dead + req.komi
    white_score = white_territory + white_dead

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
