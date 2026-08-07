"""Moku 检测器 —— ONNX Runtime 会话封装。

对齐 TS `moku-detector.ts` 的 MokuDetector，但去掉 Web Worker / 浏览器 Cache API /
模型下载进度等浏览器端逻辑，服务端只需从本地文件路径加载模型并跑推理。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .moku_postprocess import (
    DEFAULT_THRESHOLD,
    WARP_OUTPUT_SIZE,
    postprocess,
    preprocess,
)
from .types import RecognitionResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "moku-v3.onnx"


class MokuDetector:
    """封装 moku-v3 RT-DETR 模型的一次/多次推理。

    线程安全：ONNX Runtime Session 的 run() 是线程安全的，多个并发请求可
    共享同一个会话。
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    @property
    def ready(self) -> bool:
        return self.session is not None

    def detect(
        self,
        img: np.ndarray,  # HxWx4 uint8 RGBA
        board_size: int,
        threshold: float = DEFAULT_THRESHOLD,
        output_size: int = WARP_OUTPUT_SIZE,
    ) -> RecognitionResult:
        """原图 → 完整识别结果。img 必须为 uint8 RGBA。"""
        if not self.ready:
            raise RuntimeError("MokuDetector not initialized")

        tensor = preprocess(img)
        feeds = {"pixel_values": tensor}
        results = self.session.run(["logits", "pred_boxes"], feeds)
        logits, pred_boxes = results
        logits = np.asarray(logits).reshape(-1, 3)
        pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)

        return postprocess(logits, pred_boxes, img, board_size, threshold, output_size)
