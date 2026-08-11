/**
 * 领地/记分计算 —— 从前端主应用 `packages/ui/src/services/scoring.ts` 移植。
 *
 * 纯函数、无外部依赖，直接在本项目（sample_front）内运行，无需再请求后端
 * `/api/v1/score`。输入三件套（signMap + deadStones [+ probabilityMap]）已
 * 由前端本地持有，因此记分在 19×19 盘面下为亚毫秒级，同步执行、零网络往返。
 *
 * 符号约定与后端 /api/v1/score 一致：Sign 1=黑 / -1=白 / 0=空；
 * territory 沿用同一约定（-1=白地 / 0=单官中立 / 1=黑地）。
 */

/** 领地归属阈值 —— 与后端 scoring.TERRITORY_THRESHOLD=0.2 一致。 */
export const TERRITORY_THRESHOLD = 0.2;

/** 死子判定阈值 —— 与后端 deadstones.DEAD_STONE_THRESHOLD=0.4 一致。 */
export const DEAD_STONE_THRESHOLD = 0.4;

/** 坐标 key：与后端 deadStones 逐点一致；Set<string> 便于 O(1) 查重。 */
const key = (x: number, y: number): string => `${x},${y}`;

export interface TerritoryResult {
  blackTerritory: number;
  whiteTerritory: number;
  /** -1 = 白地, 0 = 中立/单官, 1 = 黑地 */
  territories: number[][];
}

/**
 * 精确路径：flood fill 找封口领地，死子先清零。对应后端 compute_territory。
 */
export function calculateTerritory(
  signMap: number[][],
  deadStones: Set<string>
): TerritoryResult {
  const height = signMap.length;
  const width = signMap[0]?.length || 0;

  // 死子清零副本
  const modifiedBoard: number[][] = signMap.map((row, y) =>
    row.map((sign, x) => (deadStones.has(key(x, y)) ? 0 : sign))
  );

  const visited = new Set<string>();
  const territories: number[][] = Array.from({ length: height }, () =>
    Array<number>(width).fill(0)
  );

  let blackTerritory = 0;
  let whiteTerritory = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (modifiedBoard[y][x] !== 0 || visited.has(key(x, y))) continue;

      const region = floodFill(modifiedBoard, [x, y], visited);
      const owner = determineOwner(modifiedBoard, region);

      for (const [vx, vy] of region) territories[vy][vx] = owner;
      if (owner === 1) blackTerritory += region.length;
      else if (owner === -1) whiteTerritory += region.length;
    }
  }

  return { blackTerritory, whiteTerritory, territories };
}

/**
 * 找与 start 连通的空点区域（栈式 DFS）。对应后端 flood_fill。
 */
function floodFill(
  board: number[][],
  start: [number, number],
  visited: Set<string>
): [number, number][] {
  const height = board.length;
  const width = board[0]?.length || 0;
  const region: [number, number][] = [];
  const stack: [number, number][] = [start];

  while (stack.length > 0) {
    const [x, y] = stack.pop()!;
    if (visited.has(key(x, y))) continue;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    if (board[y][x] !== 0) continue;

    visited.add(key(x, y));
    region.push([x, y]);
    stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
  }

  return region;
}

/**
 * 判定空域归属：只贴黑→1，只贴白→-1，两边都贴（单官）→0。对应后端 determine_owner。
 */
function determineOwner(board: number[][], region: [number, number][]): number {
  const height = board.length;
  const width = board[0]?.length || 0;

  let touchesBlack = false;
  let touchesWhite = false;

  for (const [x, y] of region) {
    if (x + 1 < width && board[y][x + 1] === 1) touchesBlack = true;
    if (x + 1 < width && board[y][x + 1] === -1) touchesWhite = true;
    if (x - 1 >= 0 && board[y][x - 1] === 1) touchesBlack = true;
    if (x - 1 >= 0 && board[y][x - 1] === -1) touchesWhite = true;
    if (y + 1 < height && board[y + 1][x] === 1) touchesBlack = true;
    if (y + 1 < height && board[y + 1][x] === -1) touchesWhite = true;
    if (y - 1 >= 0 && board[y - 1][x] === 1) touchesBlack = true;
    if (y - 1 >= 0 && board[y - 1][x] === -1) touchesWhite = true;
  }

  if (touchesBlack && !touchesWhite) return 1;
  if (touchesWhite && !touchesBlack) return -1;
  return 0; // Dane (neutral)
}

/**
 * 估计路径：flood fill 兜底 + probabilityMap 补官子。对应后端 compute_estimated_territory。
 * 先算能确定的封口领地；对 flood fill 留下的中性/单官空点，再用概率图判定：
 * prob > 0.2 → 黑地；prob < -0.2 → 白地。
 */
export function calculateEstimatedTerritory(
  signMap: number[][],
  probabilityMap: number[][],
  deadStones: Set<string>
): TerritoryResult {
  const height = signMap.length;
  const width = signMap[0]?.length || 0;

  const flood = calculateTerritory(signMap, deadStones);
  const territories = flood.territories;
  let blackTerritory = flood.blackTerritory;
  let whiteTerritory = flood.whiteTerritory;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (signMap[y][x] !== 0 || territories[y][x] !== 0) continue;

      const prob = probabilityMap[y]?.[x] ?? 0;
      if (prob > TERRITORY_THRESHOLD) {
        territories[y][x] = 1;
        blackTerritory++;
      } else if (prob < -TERRITORY_THRESHOLD) {
        territories[y][x] = -1;
        whiteTerritory++;
      }
    }
  }

  return { blackTerritory, whiteTerritory, territories };
}

/** 按子色统计死子数（死子集已给定）。对应后端 count_dead_stones。 */
export function countDeadStones(
  signMap: number[][],
  deadStones: Set<string>
): { blackDeadStones: number; whiteDeadStones: number } {
  let blackDeadStones = 0;
  let whiteDeadStones = 0;

  deadStones.forEach(k => {
    const idx = k.indexOf(',');
    const x = Number(k.slice(0, idx));
    const y = Number(k.slice(idx + 1));
    const sign = signMap[y]?.[x];
    if (sign === 1) blackDeadStones++;
    if (sign === -1) whiteDeadStones++;
  });

  return { blackDeadStones, whiteDeadStones };
}

/**
 * 计算最终分数。对应后端 /api/v1/score 的分数公式：
 *   黑方 = 黑地 + 白死子；白方 = 白地 + 黑死子 + komi。
 * capture（提子数）由对局历史提供，拍照识别场景未知，置 0。
 */
export function computeScore(
  signMap: number[][],
  deadStones: Set<string>,
  komi = 0,
  probabilityMap?: number[][]
): {
  blackTerritory: number;
  whiteTerritory: number;
  blackDeadStones: number;
  whiteDeadStones: number;
  blackScore: number;
  whiteScore: number;
  territoryMap: number[][];
} {
  const counts = countDeadStones(signMap, deadStones);
  const territory = probabilityMap
    ? calculateEstimatedTerritory(signMap, probabilityMap, deadStones)
    : calculateTerritory(signMap, deadStones);

  return {
    blackTerritory: territory.blackTerritory,
    whiteTerritory: territory.whiteTerritory,
    blackDeadStones: counts.blackDeadStones,
    whiteDeadStones: counts.whiteDeadStones,
    blackScore: territory.blackTerritory + counts.whiteDeadStones,
    whiteScore: territory.whiteTerritory + counts.blackDeadStones + komi,
    territoryMap: territory.territories,
  };
}

/**
 * 由后端 /api/v1/deadstones 响应里的死子坐标列表构造 Set<string>。
 * 入参 [{x,y}, ...]，坐标约定与后端一致（x=列, y=行）。
 */
export function deadStonesToSet(deadStones: Array<{ x: number; y: number }>): Set<string> {
  return new Set(deadStones.map(d => key(d.x, d.y)));
}
