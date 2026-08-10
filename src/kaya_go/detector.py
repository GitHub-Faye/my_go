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
from .perspective import warp_perspective
from .types import BoardCorners, MokuRawDetection, RecognitionResult

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

    def _run_inference(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """预处理 + ONNX 推理，返回 (logits, pred_boxes)。2D 展开后。"""
        tensor = preprocess(img)
        feeds = {"pixel_values": tensor}
        results = self.session.run(["logits", "pred_boxes"], feeds)
        logits = np.asarray(results[0]).reshape(-1, 3)
        pred_boxes = np.asarray(results[1]).reshape(-1, 4)
        return logits, pred_boxes

    def detect(
        self,
        img: np.ndarray,  # HxWx4 uint8 RGBA
        board_size: int,
        threshold: float = DEFAULT_THRESHOLD,
        output_size: int = WARP_OUTPUT_SIZE,
    ) -> RecognitionResult:
        """原图 → 完整识别结果。img 必须为 uint8 RGBA。

        为了与既有 /api/v1/recognize 接口的纯 result 输出保持一致，本方法
        返回完整结果但不附带原始检测与 warped 灰度图。
        """
        logits, pred_boxes = self._run_inference(img)
        result, _ = postprocess(
            logits,
            pred_boxes,
            img,
            board_size,
            threshold,
            output_size,
        )
        return result

    def detect_full(
        self,
        img: np.ndarray,  # HxWx4 uint8 RGBA
        board_size: int,
        threshold: float = DEFAULT_THRESHOLD,
        output_size: int = WARP_OUTPUT_SIZE,
        corners: BoardCorners | None = None,
    ) -> tuple[
        RecognitionResult,
        list[MokuRawDetection],
        np.ndarray,  # warped 灰度 (output_size, output_size) uint8
    ]:
        """完整识别 + 附带前后端分离所需的中间产物。

        返回 (result, raw_detections, warped_gray)：
          - result               —— 与 detect() 一致的识别结果（含默认阈值下 stones/映射）
          - raw_detections       —— 全部过阈值检出的 (class_id, score, cx, cy)，供
                                    前端本地按新阈值 re-filter（无需重跑 ONNX）
          - warped_gray          —— 透视校正后的单通道灰度小图（供前端本地重采样 /
                                    标记黑白空）。100% 对齐 result 里
                                    estimated_grid_corners 所用的 warp 空间（output_size，
                                    8% 内缩），保证前端拿 gridCorners 的坐标能直接落在
                                    warped_gray 像素上。

        当传入 corners（图像坐标 TL/TR/BR/BL）时，跳过 Moku 自动角点推断，直接用
        用户给定的角点做透视校正与网格映射（对应前端拖好 4 角再上传）。
        """
        logits, pred_boxes = self._run_inference(img)
        result, raw_detections = postprocess(
            logits,
            pred_boxes,
            img,
            board_size,
            threshold,
            output_size,
            corners=corners,
        )

        # warped 灰度小图：与 estimated_grid_corners 同一次 warp、同一坐标系。
        # 用 8% 内缩 dst（m = 0.08*output_size），网格角点即此 dst 正方形。
        m = round(output_size * 0.08)
        inset_dst: BoardCorners = (
            (m, m),
            (output_size - 1 - m, m),
            (output_size - 1 - m, output_size - 1 - m),
            (m, output_size - 1 - m),
        )
        warped_rgba = warp_perspective(img, result.corners, output_size, inset_dst)
        warped_gray = warped_rgba[..., :3].mean(axis=2).astype(np.uint8)

        return result, raw_detections, warped_gray
