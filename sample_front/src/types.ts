/** 后端 /api/v1/recognize 响应（含前后端分离新增字段）。 */
export type StoneColor = 'black' | 'white';

export interface DetectedStone {
  x: number;
  y: number;
  color: StoneColor;
}

export interface RawDetection {
  class: number;
  score: number;
  cx: number;
  cy: number;
}

export interface WarpedGray {
  width: number;
  height: number;
  dataBase64: string; // PNG base64
}

export interface RecognizeResponse {
  boardSize: number;
  stones: DetectedStone[];
  signMap: number[][];
  corners: [number, number][];
  cornersDetected: boolean;
  sgf: string;
  estimatedGridCorners: [number, number][] | null;
  mokuRawCorners: [number, number][] | null;
  mokuCornerCount: number | null;
  detections: RawDetection[];
  warpedGray: WarpedGray;
  gridCorners: [number, number][] | null;
}

/** 后端 /api/v1/deadstones 响应（Monte Carlo 死子估计）。 */
export interface DeadStonesResponse {
  boardSize: number;
  /** 领地概率图 float ∈ [-1,1]，正=黑控制、负=白控制。 */
  probabilityMap: number[][];
  deadStones: { x: number; y: number }[];
  blackDeadStones: number;
  whiteDeadStones: number;
}

/** 后端 /api/v1/corners 响应（moku 角点检测）。 */
export interface CornersResponse {
  corners: [number, number][];
  order: 'TL/TR/BR/BL';
  cornersDetected: boolean;
  /** 重建/回退时携带的 moku 原始四角（供前端对照），直接取用时可不返回 */
  mokuRawCorners?: [number, number][] | null;
  /** true = 4 角不构成近似四边形，已用可靠锚点重建仿四边形（+经典 CV 矫正） */
  rebuilt?: boolean;
}
