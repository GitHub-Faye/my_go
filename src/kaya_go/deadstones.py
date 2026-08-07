"""死子/领地估计 —— pure Python 移植自 @kaya/deadstones 的 Rust WASM 内核。

原始实现:packages/deadstones/src-rust/src/{rand.rs,pseudo_board.rs,deadstones.rs}
通过 wasm-pack 暴露为 getProbabilityMap(data, width, iterations, seed)。

本模块用 numpy 2D 棋盘替代 Rust 的扁平 Vec<Sign>,逐函数对应:
  - Rand            → rand.rs  (xorshift128)
  - PseudoBoard      → pseudo_board.rs (get_chain / has_liberties / make_pseudo_move)
  - play_till_end    → deadstones.rs
  - get_probability_map → deadstones.rs

关键约定(与 wasm 一致,勿改):
  Sign: 1=黑, -1=白, 0=空
  概率 ∈ [-1,1]: 正=黑方控制(黑地), 负=白方控制(白地)
  迭代先手平衡:前 iterations/2 局白先,后 half 局黑先
"""
from __future__ import annotations

import numpy as np

# 常量与 wasm 的 Rust 版一致
_KX = 123456789
_KY = 362436069
_KZ = 521288629
_KW = 88675123


class Rand:
    """xorshift128 PRNG —— 与 rand.rs 逐位一致(seed 推导同前端)。

    注意:range() 用 `rand().checked_rem(m)`,与 Rust `%` 保持一致的正值取模。
    """

    __slots__ = ("x", "y", "z", "w")

    def __init__(self, seed: int) -> None:
        seed &= 0xFFFFFFFF
        self.x = _KX ^ seed
        self.y = _KY ^ seed
        self.z = _KZ
        self.w = _KW

    def rand(self) -> int:
        x = self.x
        t = (x ^ (x << 11)) & 0xFFFFFFFF
        self.x, self.y, self.z = self.y, self.z, self.w
        self.w = (self.w ^ (self.w >> 19) ^ t ^ (t >> 8)) & 0xFFFFFFFF
        return self.w

    def range(self, a: int, b: int) -> int:
        m = (b - a) & 0xFFFFFFFF
        r = self.rand()
        r = r % m if m != 0 else 0  # Rust `checked_rem(m)` 当 m!=0 取模
        return a + r


class PseudoBoard:
    """2D 棋盘 + 连子/气/伪着法 —— 对应 pseudo_board.rs。"""

    __slots__ = ("data", "width", "height")

    def __init__(self, data: np.ndarray) -> None:
        # data: (H, W) int8, 值 1/-1/0
        self.data = data
        self.height, self.width = data.shape

    def clone(self) -> "PseudoBoard":
        return type(self)(self.data.copy())

    def get(self, v: tuple[int, int]) -> int:
        y, x = v
        if 0 <= y < self.height and 0 <= x < self.width:
            return int(self.data[y, x])
        raise IndexError(v)

    @staticmethod
    def _vertex_for(arr: np.ndarray, idx: int) -> tuple[int, int]:
        y, x = divmod(idx, arr.shape[1])
        return y, x

    def neighbors(self, v: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = v
        out = []
        if y > 0:
            out.append((y - 1, x))
        if y < self.height - 1:
            out.append((y + 1, x))
        if x > 0:
            out.append((y, x - 1))
        if x < self.width - 1:
            out.append((y, x + 1))
        return out


# ── 算法内核(与 deadstones.rs 逐函数对应)───────────────────────────────


def _get_chain_inner(
    arr: np.ndarray, vertex: tuple[int, int], cache: set[tuple[int, int]], sign: int
) -> list[tuple[int, int]]:
    h, w = arr.shape
    stack = [vertex]
    chain: list[tuple[int, int]] = []
    while stack:
        v = stack.pop()
        y, x = v
        if v in cache or not (0 <= y < h and 0 <= x < w):
            continue
        if arr[y, x] != sign:
            continue
        cache.add(v)
        chain.append(v)
        # 4 邻
        if y > 0:
            stack.append((y - 1, x))
        if y < h - 1:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x < w - 1:
            stack.append((y, x + 1))
    return chain


def get_chain(arr: np.ndarray, vertex: tuple[int, int]) -> list[tuple[int, int]]:
    """vertex 所在同色连子(pseudo_board.get_chain)。"""
    sign = int(arr[vertex])
    if sign == 0:
        return []
    return _get_chain_inner(arr, vertex, set(), sign)


def _has_liberties_inner(
    arr: np.ndarray, vertex: tuple[int, int], visited: set[tuple[int, int]], sign: int
) -> bool:
    h, w = arr.shape
    stack = [vertex]
    while stack:
        v = stack.pop()
        y, x = v
        if v in visited:
            continue
        if not (0 <= y < h and 0 <= x < w):
            continue
        visited.add(v)
        s = int(arr[y, x])
        if s == 0:
            return True
        if s != sign:
            continue
        if y > 0:
            stack.append((y - 1, x))
        if y < h - 1:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x < w - 1:
            stack.append((y, x + 1))
    return False


def has_liberties(arr: np.ndarray, vertex: tuple[int, int]) -> bool:
    sign = int(arr[vertex])
    if sign == 0:
        return False
    return _has_liberties_inner(arr, vertex, set(), sign)


def _get_related_chains(arr: np.ndarray, vertex: tuple[int, int]) -> list[tuple[int, int]]:
    """pseudo_board.get_related_chains: 与 vertex 共享空点的同色串(含绕空点的兄弟串)。"""
    sign = int(arr[vertex])
    if sign == 0:
        return []
    # 取跨过空点的连通(sign 或 0),再过滤只留石头
    cache: set[tuple[int, int]] = set()
    _get_chain_inner(arr, vertex, cache, sign)
    related: set[tuple[int, int]] = set()
    h, w = arr.shape
    stack = [vertex]
    visited = set()
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        y, x = v
        if not (0 <= y < h and 0 <= x < w):
            continue
        s = int(arr[y, x])
        if s != 0 and s != sign:
            continue
        if s == sign:
            related.add(v)
        for n in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            ny, nx = n
            if 0 <= ny < h and 0 <= nx < w:
                stack.append(n)
    return list(related)


def make_pseudo_move(
    arr: np.ndarray, sign: int, vertex: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """伪着法:落子 sign,提掉无气敌串。返回被提顶点列表;None=非法(自杀/无变化)。"""
    y, x = vertex
    h, w = arr.shape
    if not (0 <= y < h and 0 <= x < w):
        return None
    if arr[y, x] != 0:
        return None

    n_list = [
        v for v in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)) if 0 <= v[0] < h and 0 <= v[1] < w
    ]

    # 若所有邻点均非敌色(空或同色) → 非法(无气可占,属伪自陷判据的先行条件)
    if all(int(arr[ny, nx]) == sign or int(arr[ny, nx]) == 0 for ny, nx in n_list):
        # Rust: neighbors.iter().all(|&neighbor| s == None || s == Some(sign))
        #   注意空邻也含在内 → 直接判 None(与 Rust 一致)
        return None

    arr[y, x] = sign

    if not has_liberties(arr, (y, x)):
        # 落点无气且非单纯"点"串的情况, 需模拟提子后判断地陷自害
        # 但先走标准提子流程 (Rust 的顺序是先 set → has_liberties 检查 → 提)
        pass

    # 提子:检查 4 邻敌串
    dead: list[tuple[int, int]] = []
    capture_candidates: set[tuple[int, int]] = set()
    check_multi_dead_chains = False
    check_capture = False

    # 若落点无气 → 若非点串(周围无同色), 记 check_multi_dead_chains
    if not has_liberties(arr, (y, x)):
        is_point_chain = all(int(arr[ny, nx]) != sign for ny, nx in n_list)
        if is_point_chain:
            check_multi_dead_chains = True
        else:
            check_capture = True

    dead_chains = 0
    for ny, nx in n_list:
        if int(arr[ny, nx]) != -sign or has_liberties(arr, (ny, nx)):
            continue
        chain = get_chain(arr, (ny, nx))
        dead_chains += 1
        for c in chain:
            arr[c] = 0
            dead.append(c)

    if (check_multi_dead_chains and dead_chains <= 1) or (check_capture and len(dead) == 0):
        # 地陷反提:被提的改成敌方色(视为无气补回),复原落点
        for d in dead:
            arr[d] = -sign
        arr[y, x] = 0
        return None

    return dead


def play_till_end(arr: np.ndarray, initial_sign: int, rng: Rand) -> np.ndarray:
    """从 initial_sign 满先手随机下到双方都 pass,Rust play_till_end 对应。"""
    board = arr.copy()
    h, w = board.shape
    sign = initial_sign
    illegal_vertices: list[tuple[int, int]] = []
    finished = [False, False]
    free_vertices = [
        (y, x) for y in range(h) for x in range(w) if board[y, x] == 0
    ]

    while free_vertices and not (finished[0] and finished[1]):
        made_move = False
        while free_vertices:
            ri = rng.range(0, len(free_vertices))
            vertex = free_vertices[ri]
            del free_vertices[ri]
            freed = make_pseudo_move(board, sign, vertex)
            if freed is not None:
                free_vertices.extend(freed)
                if sign < 0:
                    finished[0] = False
                else:
                    finished[1] = False
                made_move = True
                break
            illegal_vertices.append(vertex)
        if sign > 0:
            finished[0] = not made_move
        else:
            finished[1] = not made_move
        free_vertices.extend(illegal_vertices)
        sign = -sign

    # 收尾补洞:孤立空点填给任一相邻棋子色 (与 Rust patch holes 一致)
    for y in range(h):
        for x in range(w):
            if board[y, x] != 0:
                continue
            fill = 0
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    s = int(board[ny, nx])
                    if s == 1 or s == -1:
                        fill = s
                        break
            if fill != 0:
                board[y, x] = fill

    return board


def get_probability_map(
    data: np.ndarray, iterations: int, seed: int
) -> np.ndarray:
    """Monte Carlo 领地概率图。data: (H,W) int8 → (H,W) float32 ∈ [-1,1]。

    对应 Rust get_probability_map:前后各半 iterations/2 局分由白/黑先手。
    """
    board = np.asarray(data, dtype=np.int8)
    rng = Rand(seed)
    h, w = board.shape

    neg_count = np.zeros((h, w), dtype=np.int32)  # slope 0: 白覆盖
    pos_count = np.zeros((h, w), dtype=np.int32)  # slope 1: 黑覆盖

    half = iterations // 2
    signs = [-1] * half + [1] * (iterations - half)  # 前 half 白先,后黑先

    for sign in signs:
        area = play_till_end(board, sign, rng).astype(np.int8)
        neg_count += area == -1
        pos_count += area == 1

    total = pos_count + neg_count
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(
            total == 0,
            0.0,
            (pos_count.astype(np.float32) * 2.0) / total.astype(np.float32) - 1.0,
        )
    return prob.astype(np.float32)
