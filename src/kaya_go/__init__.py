"""Kaya 围棋棋盘识别 —— FastAPI 服务端移植。

从 TS 包 @kaya/board-recognition 移植到 Python，使用 ONNX Runtime 运行
moku-v3 RT-DETR 模型进行整盘识别。

对应 TS 源码（权威参考）：
  packages/board-recognition/src/moku-postprocess.ts
  packages/board-recognition/src/moku-detector.ts
  packages/board-recognition/src/corners.ts
  packages/board-recognition/src/perspective.ts
  packages/board-recognition/src/sgf.ts
"""
