/**
 * 前后端分离样品 —— 识别 + 本地微调主流程。
 *
 * ① 网页端手动对齐 4 个角
 * ② 上传原图 + corners → 后端推理 ONNX
 * ③ 拿回结果 → 本地拖阈值 + 标记黑白空 → 得最终棋局（全部本地重算，不重传图）
 */
import { useCallback, useRef, useState, useEffect, useMemo } from 'react';
import CornerEditor from './components/CornerEditor';
import BoardPreview from './components/BoardPreview';
import { filterAndMapStones, classifyWithHints, type BoardCorners, type CalibrationHint } from './lib/geometry';
import type { RecognizeResponse, DetectedStone } from './types';

// 灵敏度滑杆（0..1）直接对应原版 Kaya 的 mokuThreshold：默认 0.965（高敏感）。
// 显示的阈值 = 1 − 灵敏度：灵敏度越高 → 阈值越低 → 显示更多棋子。
//
// 后端 /recognize 返回的 `detections` 是**全部**候选（含 score<threshold 的），
// 前端在本地按阈值 d.score >= (1 − sensitivity) 过滤显示，无需再次请求后端。
const DEFAULT_SENSITIVITY = 0.965;
// 默认滑杆（0.965）对应的显示阈值 0.035 —— 与后端 stones 的粗滤阈值一致
const DEFAULT_THRESHOLD = 1 - DEFAULT_SENSITIVITY;

type CorrState = 'black' | 'white' | 'empty' | null;

export default function App() {
  // ── ① 原图 + 角点 ──
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imgDims, setImgDims] = useState({ width: 1, height: 1 });
  const [corners, setCorners] = useState<BoardCorners | null>(null);
  const [cornersManual, setCornersManual] = useState(false);

  // ── ②③ 服务端结果 + 本地微调 ──
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [hints, setHints] = useState<Map<string, 'black' | 'white' | 'empty'>>(new Map());
  const [corrMode, setCorrMode] = useState<CorrState>(null);
  const [sensitivity, setSensitivity] = useState(DEFAULT_SENSITIVITY); // 0..1 灵敏度

  const fileInputRef = useRef<HTMLInputElement>(null);
  const warpedGrayCanvasRef = useRef<HTMLCanvasElement>(null);

  // 原图（本地 refilter / classify 需要原图做 warp？—— 用服务端返回的 corners + warpedGray 即可）
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const imageRef = useRef<HTMLImageElement | null>(null);

  // ── ① 载入图片并初始化角点（默认缩进 5% 作为初始 4 角）──
  const handleFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const w = img.width;
      const h = img.height;
      setImgDims({ width: w, height: h });
      setImageUrl(url);
      imageRef.current = img;
      const m = Math.min(w, h) * 0.05;
      setCorners([
        [m, m],
        [w - 1 - m, m],
        [w - 1 - m, h - 1 - m],
        [m, h - 1 - m],
      ]);
      setCornersManual(false);
      setResult(null);
      setHints(new Map());
    };
    img.src = url;
  }, []);

  // ── ② 上传原图 + corners → 后端 ──
  const recognize = useCallback(async () => {
    if (!imageUrl || !corners) return;
    setAnalyzing(true);
    setError(null);
    try {
      const blob = await fetch(imageUrl).then(r => r.blob());
      const fd = new FormData();
      fd.append('image', blob, 'photo.png');
      // corners 走查询串（服务端 Query 参数）：x1,y1,x2,y2,x3,y3,x4,y4（TL/TR/BR/BL）
      // —— 放进 FormData body 会被 FastAPI 吞掉，服务端收不到。
      fd.append('boardSize', '19');
      const q = corners.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(',');
      const res = await fetch(`/api/v1/recognize?corners=${encodeURIComponent(q)}`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
      }
      const body: RecognizeResponse = await res.json();
      setResult(body);
      setCornersManual(true);
      setHints(new Map());
      // 重置滑杆到默认灵敏度（0.965 → 阈值 0.035，与后端 stones 粗滤一致）
      setThreshold(DEFAULT_THRESHOLD);
      setSensitivity(DEFAULT_SENSITIVITY);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAnalyzing(false);
    }
  }, [imageUrl, corners]);

  // ── ③a 本地 refilter：按新阈值重映射 detections → stones ──
  const refinedStones = useMemo<DetectedStone[]>(() => {
    if (!result) return [];
    // 优先本地按阈值重算（无网络）；并叠加用户 via-corner 重映射
    return filterAndMapStones(result.detections, result.corners as any, result.boardSize, threshold);
  }, [result, threshold]);

  // ── ③b 本地黑白空重分类（warpedGray → 全盘 classifyWithHints）──
  const hintedStones = useMemo<DetectedStone[]>(() => {
    if (!result || !warpedGrayCanvasRef.current) return refinedStones;
    // 把 warpedGray base64 画上 canvas 取灰度数据
    const cv = warpedGrayCanvasRef.current;
    const ctx = cv.getContext('2d');
    if (!ctx) return refinedStones;

    const img = new Image();
    void img;
    // 数据已在 useEffect 里画好（见下方渲染加载）——这里保证 canvas 已填好
    // 直接读取 canvas 像素
    const data = ctx.getImageData(0, 0, cv.width, cv.height).data;
    const gray = new Float32Array(cv.width * cv.height);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = data[i * 4]; // R 通道即灰度（L 模式）
    }
    const hintList: CalibrationHint[] = [...hints.entries()].map(([key, color]) => {
      const [x, y] = key.split(',').map(Number);
      return { x, y, color };
    });
    const stones = classifyWithHints(
      { data: gray, width: cv.width, height: cv.height },
      result.boardSize,
      hintList,
      (result.gridCorners as BoardCorners) ?? undefined
    );
    return stones.length > 0 ? stones : refinedStones;
  }, [result, hints, refinedStones]);

  // 渲染时把 warpedGray 载入 canvas
  const loadWarped = useCallback(() => {
    if (!result || !warpedGrayCanvasRef.current) return;
    const img = new Image();
    img.onload = () => {
      const cv = warpedGrayCanvasRef.current!;
      const ctx = cv.getContext('2d')!;
      cv.width = result.warpedGray.width;
      cv.height = result.warpedGray.height;
      ctx.drawImage(img, 0, 0, cv.width, cv.height);
    };
    img.src = result.warpedGray.dataBase64.startsWith('data:')
      ? result.warpedGray.dataBase64
      : `data:image/png;base64,${result.warpedGray.dataBase64}`;
  }, [result]);

  // 载入结果时把 warpedGray 同步到隐藏 canvas（供本地 classify）
  useEffect(() => {
    loadWarped();
  }, [loadWarped]);

  // ── ③c 标记黑白空：点某一格 → 添加/切换 hint → 重分类 ──
  const onIntersectionClick = useCallback((col: number, row: number) => {
    if (!corrMode) return;
    setHints(prev => {
      const next = new Map(prev);
      const key = `${col},${row}`;
      // 点相同格 → 取消标记
      if (next.get(key) === corrMode) next.delete(key);
      else next.set(key, corrMode);
      return next;
    });
  }, [corrMode]);

  const resetHints = useCallback(() => {
    setHints(new Map());
    setCorrMode(null);
  }, []);

  // 最终棋盘 = 标记后的全盘分类（无标记时退化为纯 refilter）
  const finalStones = hints.size > 0 ? hintedStones : refinedStones;

  return (
    <div style={{ padding: 20, fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <h1>前后端分离棋盘识别样品</h1>
      <p style={{ color: '#555' }}>
        ① 手动对齐 4 角 → ② 上传后端推理 → ③ 本地拖阈值 / 标记黑白空
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* 左：原图 + 角点 */}
        <section>
          <h2>① 原图对齐 4 角</h2>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          {imageUrl && (
            <div style={{ width: '100%', aspectRatio: '1', border: '1px solid #ccc', marginTop: 10 }}>
              <CornerEditor
                imageUrl={imageUrl}
                width={imgDims.width}
                height={imgDims.height}
                corners={corners}
                onChange={setCorners}
              />
            </div>
          )}
          <div style={{ marginTop: 8 }}>
            <button onClick={recognize} disabled={!imageUrl || !corners || analyzing}>
              {analyzing ? '识别中…' : cornersManual ? '重新识别' : '② 上传并推理'}
            </button>
            {corners && (
              <span style={{ marginLeft: 8, fontSize: 12, color: cornersManual ? '#0a0' : '#888' }}>
                {cornersManual ? '角点已手动调整' : '点击角点拖动'}
              </span>
            )}
          </div>
        </section>

        {/* 右：对齐预览 + 微调 */}
        <section>
          <h2>③ 微调（全程本地，不上传）</h2>
          {result ? (
            <>
              <div>
                <label>
                  灵敏度 {sensitivity.toFixed(3)}（阈值 {threshold.toFixed(3)}）
                  <input
                    type="range"
                    min={0.5}
                    max={1}
                    step={0.001}
                    value={sensitivity}
                    onChange={e => {
                      const s = Number(e.target.value);
                      setSensitivity(s);
                      setThreshold(1 - s); // 灵敏度高→阈值低→显示更多棋子
                    }}
                    style={{ width: '100%' }}
                  />
                </label>
                <div style={{ marginTop: 6 }}>
                  {(['black', 'white', 'empty'] as const).map(c => (
                    <button
                      key={c}
                      onClick={() => setCorrMode(corrMode === c ? null : c)}
                      style={{ fontWeight: corrMode === c ? 'bold' : 'normal', marginRight: 6 }}
                    >
                      {c === 'black' ? '● 标黑' : c === 'white' ? '○ 标白' : '✕ 标空'}
                    </button>
                  ))}
                  <button onClick={resetHints} disabled={hints.size === 0} style={{ marginLeft: 6 }}>
                    清除标记
                  </button>
                  <span style={{ marginLeft: 8, fontSize: 12, color: '#888' }}>
                    {corrMode ? '点击右侧棋盘标记该色棋' : '棋盘点选标记'}
                  </span>
                </div>
              </div>
              <BoardPreview
                boardSize={result.boardSize}
                stones={finalStones}
                hints={hints}
                gridCorners={(result.gridCorners as BoardCorners) ?? null}
                onIntersectionClick={onIntersectionClick}
                calibrating={corrMode !== null}
                imageUrl={imageUrl}
                corners={corners}
              />
              <div style={{ marginTop: 8, fontSize: 13 }}>
                <span>检出 {finalStones.length} 子 </span>
              </div>
            </>
          ) : (
            <div style={{ color: '#999', border: '1px dashed #ccc', padding: 40, textAlign: 'center' }}>
              上传后在此显示对齐棋盘与微调控件
            </div>
          )}
          {error && <div style={{ color: '#c00', marginTop: 8 }}>⚠ {error}</div>}
        </section>
      </div>

      {/* wapr 灰度源（隐藏 canvas，供本地 classify） */}
      <canvas ref={warpedGrayCanvasRef} style={{ display: 'none' }} />
      <p style={{ marginTop: 20, fontSize: 12, color: '#999' }}>
        后端 GET /health 检查模型；本样品前端假设服务已运行在 :8000（见 vite proxy）。
      </p>
    </div>
  );
}
