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
