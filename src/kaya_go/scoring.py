"""领地/记分计算 —— 对应 packages/ui/src/services/scoring.ts。

核心逻辑逐函数移植:
  - calculateTerritory      flood fill 确定封口领地
  - floodFill / determineOwner  区域性连通与归属判定
  - calculateEstimatedTerritory  flood fill 兜底 + probabilityMap 补官子
  - countDeadStones         按子色统计死子数

符号约定与 Kaya 一致:Sign -1=白 / 0=空 / 1=黑;
territoryMap 沿用同一约定(-1=白地 / 0=单官中立 / 1=黑地)。
"""
from __future__ import annotations

# 领地归属阈值 —— 与 TS TERRITORY_THRESHOLD=0.2 一致:
# |prob| 低于该值视为接近中性,概率图不强行归属(交给 flood fill 判为单官)。
TERRITORY_THRESHOLD = 0.2


def flood_fill(board: list[list[int]], start: tuple[int, int], visited: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """找与 start 连通的空点区域(栈式 BFS/DFS)。board: signMap(死子已清零)。"""
    height = len(board)
    width = len(board[0]) if board else 0
    region: list[tuple[int, int]] = []
    stack: list[tuple[int, int]] = [start]

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if board[y][x] != 0:
            continue
        visited.add((x, y))
        region.append((x, y))
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return region


def determine_owner(board: list[list[int]], region: list[tuple[int, int]]) -> int:
    """判定空域归属:只贴黑→1,只贴白→-1,两边都贴(单官)→0。"""
    height = len(board)
    width = len(board[0]) if board else 0

    touches_black = False
    touches_white = False

    for x, y in region:
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            sign = board[ny][nx]
            if sign == 1:
                touches_black = True
            elif sign == -1:
                touches_white = True

    if touches_black and not touches_white:
        return 1
    if touches_white and not touches_black:
        return -1
    return 0  # Dame (neutral)


def compute_territory(sign_map: list[list[int]], dead_stones: set[tuple[int, int]]) -> dict:
    """精确路径:flood fill 找封口领地,死子先清零。对应 TS calculateTerritory。

    返回 {"blackTerritory": int, "whiteTerritory": int, "territories": list[list[int]]}。
    """
    height = len(sign_map)
    width = len(sign_map[0]) if sign_map else 0

    # 死子清零副本
    modified_board = [row[:] for row in sign_map]
    for x, y in dead_stones:
        if 0 <= y < height and 0 <= x < width:
            modified_board[y][x] = 0

    visited: set[tuple[int, int]] = set()
    territories = [[0] * width for _ in range(height)]
    black_territory = 0
    white_territory = 0

    for y in range(height):
        for x in range(width):
            if modified_board[y][x] == 0 and (x, y) not in visited:
                region = flood_fill(modified_board, (x, y), visited)
                owner = determine_owner(modified_board, region)
                for vx, vy in region:
                    territories[vy][vx] = owner
                if owner == 1:
                    black_territory += len(region)
                elif owner == -1:
                    white_territory += len(region)

    return {"blackTerritory": black_territory, "whiteTerritory": white_territory, "territories": territories}


def compute_estimated_territory(
    sign_map: list[list[int]],
    probability_map: list[list[float]],
    dead_stones: set[tuple[int, int]],
) -> dict:
    """估计路径:flood fill 兜底 + probabilityMap 补官子。对应 TS calculateEstimatedTerritory。

    先算能确定的封口领地;对 flood fill 留下的中性/单官空点,再用概率图判定:
    prob >  0.2 → 黑地;  prob < -0.2 → 白地。
    """
    height = len(sign_map)
    width = len(sign_map[0]) if sign_map else 0

    flood = compute_territory(sign_map, dead_stones)
    territories = flood["territories"]
    black_territory = flood["blackTerritory"]
    white_territory = flood["whiteTerritory"]

    for y in range(height):
        for x in range(width):
            if sign_map[y][x] != 0 or territories[y][x] != 0:
                continue
            row = probability_map[y] if y < len(probability_map) else None
            prob = row[x] if row is not None and x < len(row) else 0.0
            if prob > TERRITORY_THRESHOLD:
                territories[y][x] = 1
                black_territory += 1
            elif prob < -TERRITORY_THRESHOLD:
                territories[y][x] = -1
                white_territory += 1

    return {"blackTerritory": black_territory, "whiteTerritory": white_territory, "territories": territories}


def parse_dead_stones(dead_stones: list[dict]) -> set[tuple[int, int]]:
    """把死子坐标列表(来自 /api/v1/deadstones 的 [{x,y},...])转为 set[(x,y)]。"""
    return {(d["x"], d["y"]) for d in dead_stones}
