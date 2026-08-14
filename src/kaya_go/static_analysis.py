"""KataGo 静态局势估计 —— 把棋盘快照喂给引擎做单次前馈，输出静态胜率/目差。

对齐原版 Kaya 的 ONNX 引擎：
  - 输入  bin_input[1,22,H,W] + global_input[1,19]（特征编码逻辑逐条对照
    packages/ai-engine/src/onnx-featurization.ts 与 desktop onnx_engine/featurization.rs）
  - 输出  policy / value / miscvalue / ownership 四个头
  - 数值 不缩放（bin/global 直接按 0/1 写入，与 Kaya 引擎一致）

与 MCTS 的分工：这里只做 numVisits=1 的单次前馈（静态估值），不上树搜索。
更浅更快，适合"识别 → 快速读一个大方向"；若要多步推演的胜率/目差，
需另行实现树搜索（复用本模块的 featurize + run）。
"""
from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

DEFAULT_KOMI = 6.5
# 静态估值单次推理只取这几个输出头（ownership 为逐点归属热力图，可选）
_OUTPUT_NAMES = ["policy", "value", "miscvalue", "ownership"]
_LETTERS = "ABCDEFGHJKLMNOPQRST"


class StaticAnalysisError(RuntimeError):
    pass


def _compute_liberties(sign_map: np.ndarray) -> np.ndarray:
    """每点所属棋串的气数（≤4 截断）。对齐 featurization.rs compute_liberties。"""
    size = sign_map.shape[0]
    libs = np.zeros((size, size), dtype=np.int8)
    visited = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            if sign_map[y][x] == 0 or visited[y][x]:
                continue
            color = sign_map[y][x]
            # BFS 收集同串，同时统计气
            stack = [(x, y)]
            group = set()
            empty_neighbors = set()
            while stack:
                cx, cy = stack.pop()
                if visited[cy, cx]:
                    continue
                if sign_map[cy, cx] != color:
                    if sign_map[cy, cx] == 0:
                        empty_neighbors.add((cx, cy))
                    continue
                visited[cy, cx] = True
                group.add((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < size and 0 <= ny < size:
                        stack.append((nx, ny))
            lib_count = len(empty_neighbors)
            for gy, gx in group:
                libs[gy][gx] = min(lib_count, 4)
    return libs


def featurize(
    sign_map: np.ndarray,
    pla: int,  # 1=黑走, -1=白走
    komi: float = DEFAULT_KOMI,
) -> tuple[np.ndarray, np.ndarray]:
    """盘面 → KataGo 输入张量 (bin_input[1,22,H,W], global_input[1,19])。

    与 Kaya 引擎的差异：这里不传 history/ko，相关通道（6, 9-13 与 global 0-4）
    保持全 0，让模型用先验脑补 —— "静态估值"的权衡，见模块 docstring。
    """
    size = sign_map.shape[0]
    opp = -pla
    bin_input = np.zeros((1, 22, size, size), dtype=np.float32)
    libs = _compute_liberties(sign_map)

    for y in range(size):
        for x in range(size):
            bin_input[0, 0, y, x] = 1.0  # channel 0: all ones
            c = sign_map[y][x]
            if c == pla:
                bin_input[0, 1, y, x] = 1.0
            elif c == opp:
                bin_input[0, 2, y, x] = 1.0
            if c != 0:
                n = libs[y][x]
                if n == 1:
                    bin_input[0, 3, y, x] = 1.0
                elif n == 2:
                    bin_input[0, 4, y, x] = 1.0
                elif n == 3:
                    bin_input[0, 5, y, x] = 1.0

    # global[5] = selfKomi 归一化：黑走时 -komi/20，白走时 +komi/20
    global_input = np.zeros((1, 19), dtype=np.float32)
    global_input[0, 5] = (-pla * komi) / 20.0
    return bin_input, global_input


def _interpret(policy: np.ndarray, value: np.ndarray, miscvalue: np.ndarray,
               ownership: np.ndarray | None, pla: int, size: int) -> dict:
    """KataGo 四个输出头 → 黑方视角的静态指标。对齐 result_processing.rs process_raw_outputs。"""
    # winRate：value 头三分类 softmax（win/loss/unknown），取 win 类
    exp_value = np.exp(value.reshape(-1)[:3].astype(np.float64))
    winrate_current = exp_value[0] / exp_value.sum()
    black_winrate = winrate_current if pla == 1 else 1.0 - winrate_current

    # scoreLead：miscvalue 头 [scoreMean, scoreStdev, lead] 的 lead 项 ×20
    lead_current = float(miscvalue.reshape(-1)[2]) * 20.0
    black_lead = lead_current * pla

    # policy 头 → 各落点先验概率（软最大化到全棋盘合法位 + PASS）
    logits = policy.reshape(-1).astype(np.float64)
    # 该头实际输出的 move 数（= size*size+1）之外的尾部一律视为非法，取前 N 即可
    n_moves = size * size + 1
    logits = logits[:n_moves]
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # 取前 8 个候选，转 GTP 表示（PASS 特殊；x 用字母列、y 用数字行）
    order = np.argsort(-probs)[:8]
    move_suggestions = []
    for idx in order:
        if idx == size * size:
            move = "PASS"
        else:
            y, x = divmod(int(idx), size)
            move = f"{_LETTERS[x]}{size - y}"
        move_suggestions.append({"move": move, "probability": round(float(probs[idx]), 4)})

    # ownership 头 → 逐点归属（黑方视角，正=偏向黑）。shape 依 batch 展开。
    own = None
    if ownership is not None:
        own_flat = ownership.reshape(-1)[: size * size] * pla
        own = own_flat.reshape(size, size).round(2)

    return {
        "winRate": round(black_winrate, 4),
        "scoreLead": round(black_lead, 1),
        "currentTurn": "B" if pla == 1 else "W",
        "ownership": own,
        "moveSuggestions": move_suggestions,
        "visits": 1,
    }


def analyze_static(
    sign_map: list[list[int]],
    *,
    next_to_play: str | None = None,
    komi: float = DEFAULT_KOMI,
    include_ownership: bool = True,
) -> dict:
    """单次前馈静态估值。sign_map 为 boardSize×boardSize（1=黑/-1=白/0=空）。

    - next_to_play: 该谁走，'B'/'W'；省略则按子数差推断（黑先原则）。
    - 返回黑方视角 {winRate, scoreLead, currentTurn, moveSuggestions, ownership, visits, model}。

    全程 CPU、不保留会话状态；模型未就绪时抛 StaticAnalysisError。
    """
    arr = np.asarray(sign_map, dtype=np.int8)
    _validate(arr)

    if next_to_play is None:
        from kaya_go.types import derive_next_to_play

        inferred = derive_next_to_play(sign_map)
        if inferred == "unknown":
            raise StaticAnalysisError(
                "无法从子数差推断该谁走（出现提子）。请显式传入 nextToPlay='B' 或 'W'。"
            )
        next_to_play = inferred
    pla = 1 if next_to_play == "B" else -1

    session = _get_session()
    bin_input, global_input = featurize(arr, pla, komi)
    outs = session.run(_OUTPUT_NAMES, {"bin_input": bin_input, "global_input": global_input})
    policy, value, miscvalue, ownership = outs

    result = _interpret(policy, value, miscvalue, ownership if include_ownership else None, pla, arr.shape[0])
    result["model"] = str(_session_path)
    return result


def _validate(arr: np.ndarray) -> None:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise StaticAnalysisError("signMap 须为正方形矩阵")
    ok = np.isin(arr, (-1, 0, 1)).all()
    if not ok:
        raise StaticAnalysisError("signMap 只允许取值 1/-1/0")


_session: ort.InferenceSession | None = None
_session_path: str | None = None


def _get_session() -> ort.InferenceSession:
    """懒加载 KataGo ONNX 会话（CPU）。"""
    global _session, _session_path
    if _session is None:
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent.parent / "models" / "kata1-uint8.onnx"
        if not path.exists():
            raise StaticAnalysisError(
                f"KataGo 模型不存在：{path}（请将 kata1-uint8.onnx 放入 my_go/models/）"
            )
        try:
            _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _session_path = str(path)
            logger.info("KataGo 静态分析模型已加载：%s", path)
        except Exception as e:  # noqa: BLE001
            raise StaticAnalysisError(f"KataGo 模型加载失败：{e}") from e
    return _session
