"""死子/领地估计 —— 底层内核为编译到本机的 Rust (PyO3/maturin)。

原始算法移植自 @kaya/deadstones 的 Rust WASM 内核:
packages/deadstones/src-rust/src/{rand.rs,pseudo_board.rs,deadstones.rs}
本模块不再内联 PY 实现,而是把 Monte Carlo 热核(get_probability_map /
play_till_end)委托给原生扩展 `kaya_deadstones_native`(maturin 编译),
获得与原生 Rust(`opt-level=3` + `lto`)一致的性能。

关键约定(与 wasm 一致,勿改):
  Sign: 1=黑, -1=白, 0=空
  概率 ∈ [-1,1]: 正=黑方控制(黑地), 负=白方控制(白地)
  迭代先手平衡:前 iterations/2 局白先,后 half 局黑先
"""
from __future__ import annotations

import numpy as np

# 原生 Rust 内核 —— 若未执行 `maturin develop` 编译安装则导入失败,
# 给出可读提示(见 README「性能加速」小节)。
try:
    import kaya_deadstones_native as _native
except ImportError as _e:  # pragma: no cover - 环境提示
    raise ImportError(
        "未能导入 kaya_deadstones_native(Rust 死子内核)。"
        "请先运行 `maturin develop --release`(或 `uv sync --extra native`)安装。"
    ) from _e


def get_probability_map(
    data: np.ndarray, iterations: int, seed: int
) -> np.ndarray:
    """Monte Carlo 领地概率图。data: (H,W) int8 → (H,W) float32 ∈ [-1,1]。

    委托原生 Rust;与原 WASM `getProbabilityMap` 语义完全一致:
    前 iterations/2 局白先、后半黑先,逐点累加覆盖次数后映射为 p*2/n-1。
    """
    board = np.asarray(data, dtype=np.int8)
    return _native.get_probability_map(board, iterations, int(seed) & 0xFFFFFFFF)


# 阈值与前端 packages/ui/src/services/scoring.ts 的 DEAD_STONE_THRESHOLD 一致
DEAD_STONE_THRESHOLD = 0.4


def derive_dead_stones(
    probability_map: np.ndarray, sign_map: list[list[int]]
) -> list[dict[str, int]]:
    """由概率图判定死子 —— 对应前端 useScoring.deriveDeadStones。

    规则(与 TS `deriveDeadStones` 一致):
      - 黑子(sign=1)在 白方领地(prob < -threshold) → 黑死
      - 白子(sign=-1)在 黑方领地(prob > threshold) → 白死
      - 空点跳过

    返回死子坐标列表: [{x, y}, ...], 坐标沿用 DetectedStone 约定 x=列, y=行。
    本函数纯 Python 遍历,为 O(n²) 轻量操作,无需原生加速。
    """
    dead: list[dict[str, int]] = []
    for y, row in enumerate(sign_map):
        for x, sign in enumerate(row):
            if sign == 0:
                continue
            prob = float(probability_map[y, x]) if probability_map.ndim == 2 else 0.0
            if (sign == 1 and prob < -DEAD_STONE_THRESHOLD) or (
                sign == -1 and prob > DEAD_STONE_THRESHOLD
            ):
                dead.append({"x": x, "y": y})
    return dead


def count_dead_stones(
    sign_map: list[list[int]], dead_stones: list[dict[str, int]]
) -> dict[str, int]:
    """统计各色死子数 —— 对应前端 scoring.countDeadStones。

    返回 {"blackDeadStones": int, "whiteDeadStones": int}(camelCase, 对齐前端)。
    """
    black = sum(1 for d in dead_stones if sign_map[d["y"]][d["x"]] == 1)
    white = sum(1 for d in dead_stones if sign_map[d["y"]][d["x"]] == -1)
    return {"blackDeadStones": black, "whiteDeadStones": white}
