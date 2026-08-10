/**
 * 前后端分离样例 —— 本地几何 / 图像算法（纯 TS，无依赖）。
 *
 * 从 Kaya `@kaya/board-recognition` 的 TS 源码移植关键的"前端本地可复算"部分：
 *   - perspective.ts        → computeHomography / applyHomography
 *   - moku-postprocess.ts   → mapStonesToGrid（供本地拖阈值 re-filter）
 *   - stones.ts             → sampleGrid / sampleDisc / sampleVariance / kmeans3
 *                            / classifyWithHints（供本地标记黑白空）
 *
 * 目的：前端拿到后端返回的 `detections` + `warpedGray` 后，本地重算阈值与
 * 黑白空标记而无需再次上传整图 / 重跑 ONNX。
 */

// ── 类型（对齐 @kaya/board-recognition/types.ts）──
export type Point = [number, number];
export type BoardCorners = [Point, Point, Point, Point];
export type StoneColor = 'black' | 'white';
export interface DetectedStone {
  x: number; // column
  y: number; // row
  color: StoneColor;
}

// 后端 /api/v1/recognize 返回的 detections 元素
export interface RawDetection {
  class: number; // 0=黑子 1=白子 2=角点
  score: number;
  cx: number;
  cy: number;
}

export const CLASS_BLACK_STONE = 0;
export const CLASS_WHITE_STONE = 1;

// ── 单应矩阵（perspective.ts 移植）──
const EPS = 1e-12;

function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < EPS) return null;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let k = col; k < n + 1; k++) M[r][k] -= f * M[col][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

export function computeHomography(
  src: [Point, Point, Point, Point],
  dst: [Point, Point, Point, Point]
): number[] | null {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = src[i];
    const [dx, dy] = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
    b.push(dx);
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    b.push(dy);
  }
  const h = solveLinear(A, b);
  if (!h) return null;
  return [...h, 1.0];
}

export function applyHomography(H: number[], x: number, y: number): Point {
  const w = H[6] * x + H[7] * y + H[8];
  if (Math.abs(w) < EPS) return [x, y];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

// ── mapStonesToGrid（moku-postprocess.ts 移植）──
/** 把过阈值检出按新阈值过滤后映射到离散棋盘交叉点。score 降序占位去重。 */
export function filterAndMapStones(
  detections: RawDetection[],
  corners: BoardCorners,
  boardSize: number,
  threshold: number
): DetectedStone[] {
  // 只取黑/白（class 0/1）且 score ≥ threshold 的检测
  const stones = detections
    .filter(d => (d.class === CLASS_BLACK_STONE || d.class === CLASS_WHITE_STONE) && d.score >= threshold)
    .map(d => ({ ...d }));
  if (stones.length === 0) return [];

  const dst: [Point, Point, Point, Point] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  const H = computeHomography(corners, dst);
  if (!H) return [];

  stones.sort((a, b) => b.score - a.score); // 高分优先占位
  const result: DetectedStone[] = [];
  const occupied = new Set<string>();
  for (const det of stones) {
    const [rx, ry] = applyHomography(H, det.cx, det.cy);
    const col = Math.round(rx * (boardSize - 1));
    const row = Math.round(ry * (boardSize - 1));
    if (col < 0 || col >= boardSize || row < 0 || row >= boardSize) continue;
    const key = `${col},${row}`;
    if (occupied.has(key)) continue;
    occupied.add(key);
    result.push({ x: col, y: row, color: det.class === CLASS_BLACK_STONE ? 'black' : 'white' });
  }
  return result;
}

// ── classic CV：sampleGrid + classifyWithHints（stones.ts 移植）──
function sampleDisc(
  data: Float32Array,
  cx: number,
  cy: number,
  radius: number,
  width: number,
  height: number
): number {
  const r2 = radius * radius;
  const x0 = Math.max(0, Math.ceil(cx - radius));
  const x1 = Math.min(width - 1, Math.floor(cx + radius));
  const y0 = Math.max(0, Math.ceil(cy - radius));
  const y1 = Math.min(height - 1, Math.floor(cy + radius));
  let sum = 0;
  let count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) {
        sum += data[y * width + x];
        count++;
      }
    }
  }
  return count > 0 ? sum / count : 0;
}

function sampleVariance(
  data: Float32Array,
  cx: number,
  cy: number,
  radius: number,
  width: number,
  height: number
): number {
  const r2 = radius * radius;
  const x0 = Math.max(0, Math.ceil(cx - radius));
  const x1 = Math.min(width - 1, Math.floor(cx + radius));
  const y0 = Math.max(0, Math.ceil(cy - radius));
  const y1 = Math.min(height - 1, Math.floor(cy + radius));
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) {
        const v = data[y * width + x];
        sum += v;
        sumSq += v * v;
        count++;
      }
    }
  }
  if (count < 2) return 0;
  const mean = sum / count;
  return Math.sqrt(Math.max(0, sumSq / count - mean * mean));
}

function kmeans3(values: number[], maxIter = 20): [number, number, number] {
  if (values.length < 3) return [0, 0, 0];
  const sorted = [...values].sort((a, b) => a - b);
  let c0 = sorted[Math.floor(sorted.length * 0.1)];
  let c1 = sorted[Math.floor(sorted.length * 0.5)];
  let c2 = sorted[Math.floor(sorted.length * 0.9)];
  for (let iter = 0; iter < maxIter; iter++) {
    let s0 = 0,
      s1 = 0,
      s2 = 0;
    let n0 = 0,
      n1 = 0,
      n2 = 0;
    for (const v of values) {
      const d0 = Math.abs(v - c0);
      const d1 = Math.abs(v - c1);
      const d2 = Math.abs(v - c2);
      if (d0 <= d1 && d0 <= d2) {
        s0 += v;
        n0++;
      } else if (d1 <= d2) {
        s1 += v;
        n1++;
      } else {
        s2 += v;
        n2++;
      }
    }
    const newC0 = n0 > 0 ? s0 / n0 : c0;
    const newC1 = n1 > 0 ? s1 / n1 : c1;
    const newC2 = n2 > 0 ? s2 / n2 : c2;
    if (Math.abs(newC0 - c0) + Math.abs(newC1 - c1) + Math.abs(newC2 - c2) < 0.5) break;
    c0 = newC0;
    c1 = newC1;
    c2 = newC2;
  }
  const cs = [c0, c1, c2].sort((a, b) => a - b);
  return [cs[0], cs[1], cs[2]];
}

interface GrayImage {
  data: Float32Array;
  width: number;
  height: number;
}

/** 栅格交叉点采样：亮度 + 局部相对亮度（相对 ±3 邻域中位数）。 */
export function sampleGridFromGray(
  gray: GrayImage,
  boardSize: number,
  gridCorners?: BoardCorners
): { brightness: Float32Array; relative: Float32Array; variances: Float32Array } {
  const { data, width, height } = gray;
  let cellSize: number;
  if (gridCorners) {
    const [tl, tr, , bl] = gridCorners;
    const gridW = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
    const gridH = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
    cellSize = (gridW + gridH) / (2 * (boardSize - 1));
  } else {
    cellSize = (width - 1) / (boardSize - 1);
  }
  const discRadius = cellSize * 0.35;
  const varRadius = cellSize * 0.35;
  const N = boardSize * boardSize;
  const brightness = new Float32Array(N);
  const variances = new Float32Array(N);
  void N;

  for (let row = 0; row < boardSize; row++) {
    for (let col = 0; col < boardSize; col++) {
      let cx: number;
      let cy: number;
      if (gridCorners) {
        const u = col / (boardSize - 1);
        const v = row / (boardSize - 1);
        const [tl, tr, br, bl] = gridCorners;
        cx = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0];
        cy = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1];
      } else {
        cx = col * cellSize;
        cy = row * cellSize;
      }
      const idx = row * boardSize + col;
      brightness[idx] = sampleDisc(data, cx, cy, discRadius, width, height);
      variances[idx] = sampleVariance(data, cx, cy, varRadius, width, height);
    }
  }

  const RING = 3;
  const relative = new Float32Array(N);
  for (let row = 0; row < boardSize; row++) {
    for (let col = 0; col < boardSize; col++) {
      const neighbors: number[] = [];
      for (let dr = -RING; dr <= RING; dr++) {
        for (let dc = -RING; dc <= RING; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = row + dr;
          const nc = col + dc;
          if (nr >= 0 && nr < boardSize && nc >= 0 && nc < boardSize) {
            neighbors.push(brightness[nr * boardSize + nc]);
          }
        }
      }
      neighbors.sort((a, b) => a - b);
      const localMedian = neighbors[Math.floor(neighbors.length / 2)];
      relative[row * boardSize + col] = brightness[row * boardSize + col] - localMedian;
    }
  }
  return { brightness, relative, variances };
}

export interface CalibrationHint {
  x: number;
  y: number;
  color: StoneColor | 'empty';
}

/** 用用户标记（黑/白/空）重新分类全盘。hints=[{}] 等效于无标记的纯 k-means。 */
export function classifyWithHints(
  gray: GrayImage,
  boardSize: number,
  hints: CalibrationHint[],
  gridCorners?: BoardCorners
): DetectedStone[] {
  const { relative, variances } = sampleGridFromGray(gray, boardSize, gridCorners);
  const hintMap = new Map<string, CalibrationHint>();
  const blackVals: number[] = [];
  const whiteVals: number[] = [];
  const emptyVals: number[] = [];

  for (const h of hints) {
    hintMap.set(`${h.x},${h.y}`, h);
    const v = relative[h.y * boardSize + h.x];
    if (h.color === 'black') blackVals.push(v);
    else if (h.color === 'white') whiteVals.push(v);
    else emptyVals.push(v);
  }

  const relValues = Array.from(relative);
  const [kmBlack, kmBoard, kmWhite] = kmeans3(relValues);
  const blackC = blackVals.length > 0 ? blackVals.reduce((a, b) => a + b, 0) / blackVals.length : kmBlack;
  const boardC = emptyVals.length > 0 ? emptyVals.reduce((a, b) => a + b, 0) / emptyVals.length : kmBoard;
  const whiteC = whiteVals.length > 0 ? whiteVals.reduce((a, b) => a + b, 0) / whiteVals.length : kmWhite;

  const blackBoundary = (blackC + boardC) / 2;
  const whiteBoundary = (boardC + whiteC) / 2;
  const sortedVar = Array.from(variances).sort((a, b) => a - b);
  const medianVar = sortedVar[Math.floor(sortedVar.length / 2)];
  const totalSpread = whiteC - blackC;
  const hasBlack = totalSpread > 2 && boardC - blackC > totalSpread * 0.1;
  const hasWhite = totalSpread > 2 && whiteC - boardC > totalSpread * 0.1;

  const stones: DetectedStone[] = [];
  for (let row = 0; row < boardSize; row++) {
    for (let col = 0; col < boardSize; col++) {
      const key = `${col},${row}`;
      const hint = hintMap.get(key);
      if (hint) {
        if (hint.color !== 'empty') stones.push({ x: col, y: row, color: hint.color as StoneColor });
        continue;
      }
      const idx = row * boardSize + col;
      const r = relative[idx];
      const highVar = variances[idx] > medianVar * 3;
      if (hasBlack && r < blackBoundary) {
        if (!highVar || r < blackC * 0.5) stones.push({ x: col, y: row, color: 'black' });
      } else if (hasWhite && r > whiteBoundary) {
        if (!highVar || r > whiteC * 0.5) stones.push({ x: col, y: row, color: 'white' });
      }
    }
  }
  return stones;
}
