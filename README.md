# my_go — Kaya 棋盘识别 FastAPI 服务

把 Kaya monorepo 中 `@kaya/board-recognition` 的「识别照片 → 棋盘 SGF」能力
作为独立 HTTP 服务暴露。推理基于 ONNX Runtime + Moku v3 RT-DETR 模型，
移植自 TS 版 `MokuDetector.detect`。死子/领地估计用原生 Rust 内核
（PyO3/maturin 编译）获得与 Kaya 原生 WASM 一致的性能。

## 快速开始

```bash
uv sync                         # 安装依赖

# 死子内核需要先把 Rust 扩展编译进虚拟环境（= Kaya 的 wasm-pack 步骤）
uv run maturin develop --release --manifest-path src/kaya_go/_deadstones_rs/Cargo.toml

cp ../../apps/desktop/public/models/moku-v3.onnx models/
uv run uvicorn main:app --reload --port 8000
```

交互式文档：http://localhost:8000/docs

## 接口

### `POST /api/v1/recognize`

multipart 上传，识别一张围棋棋盘照片。

| 字段        | 类型    | 说明                                     |
| ----------- | ------- | ---------------------------------------- |
| `image`     | 文件    | PNG/JPEG/WebP 等（Pillow 支持）           |
| `boardSize` | int     | 9 / 13 / 19，默认 19                     |
| `threshold` | float   | 石头置信度，默认 0.035                   |

响应（对齐浏览器端 `RecognitionResult` 的 JSON 形态）：

```jsonc
{
  "boardSize": 19,
  "stones": [{ "x": 15, "y": 4, "color": "black" }],
  "signMap": [/* boardSize×boardSize 矩阵：1=黑 / -1=白 / 0=空 */],
  "corners": [[31, 30], [671, 30], [670, 669], [30, 670]],
  "cornersDetected": true,
  "sgf": "(;GM[1]FF[4]SZ[19]AP[Kaya Board Recognition]AB[pe][dd]\n)",
  "estimatedGridCorners": null,
  "mokuRawCorners": null,
  "mokuCornerCount": 4
}
```

`corners` 为图像坐标系中棋盘四角 [TL, TR, BR, BL]。`signMap` 可直接作为
`/api/v1/deadstones` 的输入。

### `POST /api/v1/deadstones`

`signMap → 死子状况`，Monte Carlo 领地估计后按前端阈值判定死子。

| 字段         | 类型                      | 说明                                     |
| ------------ | ------------------------- | ---------------------------------------- |
| `signMap`    | int[][]  (9/13/19 方阵)  | 1=黑 / -1=白 / 0=空（来自 recognize 响应）|
| `iterations` | int    默认 2000        | Monte Carlo 迭代次数（前一半白先、后一半黑先）|
| `seed`       | int?   默认当前时间      | 随机种子                                   |

响应：

```jsonc
{
  "boardSize": 9,
  "probabilityMap": [[/* float ∈ [-1,1]，正=黑控制、负=白控制 */]],
  "deadStones": [{ "x": 3, "y": 4 }],
  "blackDeadStones": 5,
  "whiteDeadStones": 2
}
```

### `GET /health`

```jsonc
{ "status": "ok", "model_loaded": true, "model_exists": true }
```

## 模型

- 默认从 `models/moku-v3.onnx` 加载（已在 `.gitignore` 中排除，请自行放置或从
  `apps/desktop/public/models/` 复制）。
- 也可用环境变量指定：`KAYA_MOKU_MODEL=/path/to/model.onnx`。

## 目录结构

```
my_go/
├── main.py                    # FastAPI 应用 + 路由
├── src/kaya_go/
│   ├── detector.py            # MokuDetector（ONNX Runtime 会话）
│   ├── moku_postprocess.py    # 前/后处理（RT-DETR 解码、角点、网格映射）
│   ├── corners.py             # 角点排序/退化处理
│   ├── perspective.py         # 单应矩阵 + 透视校正
│   ├── sgf.py                 # SGF 生成
│   ├── deadstones.py          # 死子/领地估计（薄封装，委托原生 Rust 内核）
│   ├── types.py               # 数据模型
│   └── _deadstones_rs/        # Rust 原生内核（PyO3/maturin）
│       ├── Cargo.toml
│       └── src/lib.rs         # 移植自 packages/deadstones/src-rust/*
└── tests/test_recognition.py  # 端到端 + 直接调用测试
```

## 测试

```bash
uv run pytest
```

## 性能加速：死子内核用 Rust

纯 Python 逐格 Monte Carlo 在 19 路、默认 2000 次迭代下非常慢
（且早期版本因移值语义误用存在无界重试）。现把热核移交给编译后的
Rust（`kaya_deadstones_native`，见 `src/kaya_go/_deadstones_rs/`），
逐函数移植自 `packages/deadstones/src-rust/src/{rand.rs,pseudo_board.rs,deadstones.rs}`：

- `Rand` —— xorshift128 PRNG，与原版逐位一致
- `PseudoBoard` —— 扁平 `Vec<Sign>` + 宽，`get_neighbors` 用 `wrapping_sub` 复用边界槽
- `make_pseudo_move` / `play_till_end` / `get_probability_map` —— 语义与 WASM 版一致
  （包含前一半白先、后一半黑先的迭代平衡）

实测（本机 19 路空盘，2000 次迭代）约 **0.9s**；9 路约 **0.14s**，与 Kaya 原生相当。

若未编译 Rust 扩展即调用死子接口，会在 `import` 阶段抛出带提示的 `ImportError`，
提示运行 `maturin develop --release`。

## 移植说明

逻辑逐函数对应 TS 源（`packages/board-recognition/src/*`）：

- `moku-postprocess.ts` → `moku_postprocess.py`
- `moku-detector.ts` → `detector.py`（去掉 Worker/缓存/下载进度）
- `corners.ts` → `corners.py`
- `perspective.ts` → `perspective.py`
- `sgf.ts` → `sgf.py`

服务端不返回 `warpedImage`（浏览器端用于拖拽角点预览，HTTP 场景无此需求）。

关键差异：
- **预处理**：TS 用 `onnxruntime-web` 的 Tensor，Python 用 numpy CHW，像素仅 `/255`（模型 `do_normalize=false`），已核对。
- **线程安全**：ONNX Runtime Session 的 `run()` 线程安全，全局共享单例，无需加锁。
- **死子内核**：Rust 编译为本地扩展（见上文），而非浏览器 WASM。

