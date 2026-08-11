import { useEffect, useMemo, useRef, useState } from 'react';
import { computeHomography, type Point, type BoardCorners } from '../lib/geometry';
import type { DetectedStone } from '../types';

interface BoardPreviewProps {
  boardSize: number;
  stones: DetectedStone[];
  hints: Map<string, 'black' | 'white' | 'empty'>;
  gridCorners: BoardCorners | null;
  onIntersectionClick: (col: number, row: number) => void;
  calibrating: boolean;
  /** 原始彩色照片 objectURL —— 作为棋盘背景（GPU 透视变换），叠加上层标记。 */
  imageUrl: string | null;
  /** 用户拖好的 4 角（图像坐标 TL/TR/BR/BL）—— 用于 CSS matrix3d 透视背景。 */
  corners: BoardCorners | null;
  /**
   * 领地覆盖层：1=黑地 / -1=白地（0 不画），死子标 ×。
   * 启用后在棋子下方铺半透明色块即可区分死子与领地。
   */
  territoryMap?: number[][] | null;
  /** 被标为死子的坐标（集合 key x,y），绘制 × 标记。 */
  deadStoneSet?: Set<string>;
}

// warped 统一空间固定尺寸（后端约定 800）—— 网格角点数值直接落在该空间
export const WARP_SIZE = 800;
// 与原版 Kaya 一致：棋盘内缩 margin（8%）
const WARP_MARGIN = 0.08;

function gridToCanvas(
  col: number,
  row: number,
  boardSize: number,
  scale: number,
  gridCorners: BoardCorners | null
): Point {
  if (gridCorners) {
    const [tl, tr, br, bl] = gridCorners;
    const u = col / (boardSize - 1);
    const v = row / (boardSize - 1);
    return [
      ((1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0]) * scale,
      ((1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1]) * scale,
    ];
  }
  const cellSize = ((WARP_SIZE - 1) / (boardSize - 1)) * scale;
  return [col * cellSize, row * cellSize];
}

/**
 * 对齐后的棋盘预览：叠加网格、检出棋子、用户标记（黑/白/空）。
 *
 * 对齐原版 Kaya `BoardPreview` 的做法：
 *  - 观察父容器，取 `min(width, height)` 作为正方形 containerSize（不依赖固定值）。
 *  - 原始彩色照片用 CSS `matrix3d`（由 corners → containerSize 内缩方块算出的单应）
 *    GPU 透视变换铺满容器，不灰度化、不留边、不裁切。
 *  - 叠加 canvas 随 containerSize 重置尺寸，`scale = containerSize / WARP_SIZE`，
 *    网格角点据此精确对齐背景原图。
 */
export default function BoardPreview({
  boardSize,
  stones,
  hints,
  gridCorners,
  onIntersectionClick,
  calibrating,
  imageUrl,
  corners,
  territoryMap,
  deadStoneSet,
}: BoardPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [containerSize, setContainerSize] = useState(0);

  // 观察父容器取较小边作为正方形尺寸，避免反馈回路
  useEffect(() => {
    const parent = containerRef.current?.parentElement;
    if (!parent) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      const s = Math.min(width, height);
      if (s > 0) setContainerSize(s);
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  // CSS matrix3d 用于图片透视变换。dst 是基于 containerSize 的内缩方块（非固定值），
  // 这样变换后的棋盘正好贴满容器四边。
  const cssTransform = useMemo(() => {
    if (!corners || !containerSize) return '';
    const m = containerSize * WARP_MARGIN;
    const dstCorners: [Point, Point, Point, Point] = [
      [m, m],
      [containerSize - 1 - m, m],
      [containerSize - 1 - m, containerSize - 1 - m],
      [m, containerSize - 1 - m],
    ];
    const H = computeHomography(corners, dstCorners);
    if (!H) return '';
    // 3×3 单应 → CSS matrix3d（列主序 4×4）
    return `matrix3d(${H[0]},${H[3]},0,${H[6]},${H[1]},${H[4]},0,${H[7]},0,0,1,0,${H[2]},${H[5]},0,${H[8]})`;
  }, [corners, containerSize]);

  // 叠加 canvas：随 containerSize 重置尺寸，scale 对齐网格角点与背景
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !containerSize) return;
    const size = containerSize;
    if (cv.width !== size || cv.height !== size) {
      cv.width = size;
      cv.height = size;
    }
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const scale = size / WARP_SIZE;
    ctx.clearRect(0, 0, size, size);

    // 网格（浅色，叠在彩色底图上仍清晰）
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < boardSize; i++) {
      const [x0, y0] = gridToCanvas(0, i, boardSize, scale, gridCorners);
      const [x1, y1] = gridToCanvas(boardSize - 1, i, boardSize, scale, gridCorners);
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      const [x2, y2] = gridToCanvas(i, 0, boardSize, scale, gridCorners);
      const [x3, y3] = gridToCanvas(i, boardSize - 1, boardSize, scale, gridCorners);
      ctx.moveTo(x2, y2);
      ctx.lineTo(x3, y3);
    }
    ctx.stroke();

    // 领地覆盖层（死子图与领地合计在一张图）：半透明色铺在每格交点处。
    // 领地 1=黑(蓝-green 半透明实心块) / -1=白(橙)，画在棋子下方 → 死子仍可见。
    const cellPx = ((WARP_SIZE - 1) / (boardSize - 1)) * scale;
    const r = Math.max(3, cellPx * 0.3);
    if (territoryMap) {
      for (let y = 0; y < boardSize; y++) {
        for (let x = 0; x < boardSize; x++) {
          const sign = territoryMap[y]?.[x];
          if (sign !== 1 && sign !== -1) continue;
          const [cx, cy] = gridToCanvas(x, y, boardSize, scale, gridCorners);
          // 居中铺在交叉点上（与棋子同中心，色块比棋子略大）
          const br = cellPx * 0.5;
          ctx.beginPath();
          ctx.fillStyle = sign === 1 ? 'rgba(40, 120, 210, 0.28)' : 'rgba(255, 90, 40, 0.28)';
          ctx.fillRect(cx - br, cy - br, br * 2, br * 2);
        }
      }
    }

    // 检出棋子：半透明色块（对齐原版 Kaya 配色 —— 黑=透明蓝、白=透明橙）。
    for (const s of stones) {
      const [cx, cy] = gridToCanvas(s.x, s.y, boardSize, scale, gridCorners);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = s.color === 'black' ? 'rgba(0, 180, 255, 0.55)' : 'rgba(255, 80, 0, 0.55)';
      ctx.fill();
    }

    // 用户手标（与检出棋子同规格的透明点）：标黑→蓝点、标白→红点、标空→去掉该点
    for (const [key, color] of hints) {
      if (color === 'empty') continue; // 空标记由全盘重分类把该点去掉，无需绘制
      const [cx0, cy0] = key.split(',').map(Number);
      const [cx, cy] = gridToCanvas(cx0, cy0, boardSize, scale, gridCorners);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = color === 'black' ? 'rgba(0, 180, 255, 0.85)' : 'rgba(255, 80, 0, 0.85)';
      ctx.fill();
      // 白色细环，与模型检出点区分，提示这是用户手标
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // 死子 × 标记：叠加在最上层（不影响领地色块）
    if (deadStoneSet) {
      const crossR = Math.max(4, cellPx * 0.2);
      for (const key of deadStoneSet) {
        const [x, y] = key.split(',').map(Number);
        if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) continue;
        const [cx, cy] = gridToCanvas(x, y, boardSize, scale, gridCorners);
        ctx.strokeStyle = 'rgba(220, 40, 40, 0.95)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - crossR, cy - crossR);
        ctx.lineTo(cx + crossR, cy + crossR);
        ctx.moveTo(cx + crossR, cy - crossR);
        ctx.lineTo(cx - crossR, cy + crossR);
        ctx.stroke();
      }
    }
  }, [boardSize, stones, hints, gridCorners, containerSize, territoryMap, deadStoneSet]);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!calibrating) return;
    const cv = canvasRef.current!;
    const rect = cv.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (cv.width / rect.width);
    const my = (e.clientY - rect.top) * (cv.height / rect.height);
    const scale = cv.width / WARP_SIZE;

    let bestDist = Infinity;
    let bestCol = 0;
    let bestRow = 0;
    for (let row = 0; row < boardSize; row++) {
      for (let col = 0; col < boardSize; col++) {
        const [gx, gy] = gridToCanvas(col, row, boardSize, scale, gridCorners);
        const d = Math.hypot(mx - gx, my - gy);
        if (d < bestDist) {
          bestDist = d;
          bestCol = col;
          bestRow = row;
        }
      }
    }
    if (bestCol >= 0 && bestCol < boardSize && bestRow >= 0 && bestRow < boardSize) {
      onIntersectionClick(bestCol, bestRow);
    }
  };

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: containerSize > 0 ? containerSize : '100%',
        height: containerSize > 0 ? containerSize : 'auto',
        aspectRatio: '1',
        margin: '0 auto',
        background: '#d8c9a8',
        overflow: 'hidden',
        borderRadius: 4,
      }}
    >
      {imageUrl && cssTransform && (
        <img
          src={imageUrl}
          alt=""
          draggable={false}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            transform: cssTransform,
            transformOrigin: '0 0',
            pointerEvents: 'none',
            backfaceVisibility: 'hidden',
            maxWidth: 'none',
          }}
        />
      )}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          cursor: calibrating ? 'crosshair' : 'default',
          touchAction: 'none',
        }}
        onClick={onClick}
      />
    </div>
  );
}
