import { useEffect, useRef, useState } from 'react';
import type { BoardCorners, Point } from '../lib/geometry';

interface CornerEditorProps {
  imageUrl: string; // objectURL of loaded image
  width: number;
  height: number;
  corners: BoardCorners | null;
  onChange: (corners: BoardCorners) => void;
}

const CORNER_COLORS = ['#00e5ff', '#ff4081', '#76ff03', '#ffd740'];
const CORNER_LABELS = ['TL', 'TR', 'BR', 'BL'];
const HANDLE_R = 14;
const HIT_R = 28;

/**
 * 原图 canvas：显示图片 + 4 个可拖动角点。角点坐标在 image 像素空间。
 * 参照 Kaya `PhotoPanel` + `useCanvasInteraction`。
 */
export default function CornerEditor({
  imageUrl,
  width,
  height,
  corners,
  onChange,
}: CornerEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const dragOffset = useRef<Point>([0, 0]);

  // 绘制主循环
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    // 让 canvas 等于图片像素，CSS 再缩放到容器
    cv.width = width;
    cv.height = height;
    const ctx = cv.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, width, height);
      drawCorners(ctx);
    };
    img.src = imageUrl;
    // 每帧重绘角点
    return () => {
      img.onload = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl, width, height, corners]);

  const toImageCoords = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const cv = canvasRef.current!;
    const rect = cv.getBoundingClientRect();
    const sx = width / rect.width;
    const sy = height / rect.height;
    return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
  };

  function drawCorners(ctx: CanvasRenderingContext2D) {
    if (!corners) return;
    for (let i = 0; i < 4; i++) {
      const [cx, cy] = corners[i];
      const color = CORNER_COLORS[i];

      ctx.beginPath();
      ctx.arc(cx, cy, HANDLE_R, 0, Math.PI * 2);
      ctx.fillStyle = color + '30';
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(cx - 10, cy);
      ctx.lineTo(cx + 10, cy);
      ctx.moveTo(cx, cy - 10);
      ctx.lineTo(cx, cy + 10);
      ctx.stroke();

      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 0.8;
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(CORNER_LABELS[i], cx, cy - 16);
    }
  }

  // 指针交互（拖角）
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!corners) return;
    const [mx, my] = toImageCoords(e);
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < 4; i++) {
      const [cx, cy] = corners[i];
      const d = Math.hypot(mx - cx, my - cy);
      if (d < HIT_R && d < bestDist) {
        best = i;
        bestDist = d;
      }
    }
    if (best >= 0) {
      setDragging(best);
      dragOffset.current = [mx - corners[best][0], my - corners[best][1]];
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragging === null || !corners) return;
    const [mx, my] = toImageCoords(e);
    const updated = [...corners] as BoardCorners;
    updated[dragging] = [mx - dragOffset.current[0], my - dragOffset.current[1]];
    onChange(updated);
  };

  const onPointerUp = () => setDragging(null);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        cursor: corners ? 'crosshair' : 'default',
        touchAction: 'none',
        display: 'block',
        background: '#000',
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    />
  );
}
