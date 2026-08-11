/**
 * 前后端分离样品 —— 识别 + 本地微调主流程。
 *
 * ① 上传原图 → 可选「自动识别 4 角」（调后端 /api/v1/corners，用 moku 模型）
 *    → 后端返回 4 角点，前端标注在原图上（可继续手动拖动微调）
 * ② 上传原图 + corners → 后端推理 ONNX
 * ③ 拿回结果 → 本地拖阈值 + 标记黑白空 → 得最终棋局（全部本地重算，不重传图）
 */
import { useCallback, useRef, useState, useMemo } from 'react';
import CornerEditor from './components/CornerEditor';
import BoardPreview from './components/BoardPreview';
import ScoreDisplay, { type ScoreBreakdown } from './components/ScoreDisplay';
import { filterAndMapStones, type DetectedStone, type BoardCorners } from './lib/geometry';
import { computeScore, deadStonesToSet } from './lib/scoring';
import type {
  RecognizeResponse,
  CornersResponse,
  DeadStonesResponse,
  StoneColor,
} from './types';

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
  const [detectingCorners, setDetectingCorners] = useState(false);

  // ── ②③ 服务端结果 + 本地微调 ──
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [hints, setHints] = useState<Map<string, 'black' | 'white' | 'empty'>>(new Map());
  const [corrMode, setCorrMode] = useState<CorrState>(null);
  const [sensitivity, setSensitivity] = useState(DEFAULT_SENSITIVITY); // 0..1 灵敏度

  // ── 死子估计 + 记分 ──
  const [deadstones, setDeadstones] = useState<DeadStonesResponse | null>(null);
  const [computingDeadstones, setComputingDeadstones] = useState(false);
  const KOMI = 6.5; // 白方贴目（与后端默认一致）

  const fileInputRef = useRef<HTMLInputElement>(null);

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
      setDeadstones(null);
    };
    img.src = url;
  }, []);

  // ── ①b 自动识别 4 角：上传原图 → 后端 /api/v1/corners（moku 角点检测）──
  const detectCorners = useCallback(async () => {
    if (!imageUrl) return;
    setDetectingCorners(true);
    setError(null);
    try {
      const blob = await fetch(imageUrl).then(r => r.blob());
      const fd = new FormData();
      fd.append('image', blob, 'photo.png');
      const res = await fetch('/api/v1/corners', {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
      }
      const body: CornersResponse = await res.json();
      // 后端返回 [x,y]×4（TL/TR/BR/BL），直接用于 CornerEditor。
      // Moku 已按顺时针排好序；直接用即可。
      const detected = body.corners as BoardCorners;
      // rebuilt=true 表示 Moku 四角不构成近似四边形，已用可靠锚点重建仿四边形
      // 并交给经典 CV 矫正——此时提示用户检查四角对齐。
      if (body.rebuilt) {
        setError(
          'Moku 检出的四角不构成近似四边形，已用可靠锚点重建并交给经典 CV 矫正。请检查四角对齐，必要时手动拖动微调。',
        );
      }
      setCorners(detected);
      setCornersManual(true);
    } catch (e) {
      setError(`自动识别角点失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDetectingCorners(false);
    }
  }, [imageUrl]);

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
      setDeadstones(null); // 新局面 → 清空旧死子/记分
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

  // ── ③c' 手动标记 = 本地强制覆盖（仅改被点的那一格，其它格不动）──
  // 用户标黑/标白/标空只叠加在细化结果之上：
  //  - 标黑/标白 → 该格强制为对应颜色（不管模型原来怎么判）
  //  - 标空     → 从结果里去掉该格
  // 不再走全盘重分类（classifyWithHints），避免一个标记带动其它格子变色。
  const finalizedStones = useMemo<DetectedStone[]>(() => {
    if (hints.size === 0) return refinedStones;
    const overridden = new Map<string, StoneColor | 'empty'>();
    for (const [key, color] of hints) overridden.set(key, color);
    const out = refinedStones.filter(s => !overridden.has(`${s.x},${s.y}`) || overridden.get(`${s.x},${s.y}`) === s.color);
    for (const [key, color] of overridden) {
      if (color === 'empty') continue; // 去掉该格
      const [x, y] = key.split(',').map(Number);
      out.push({ x, y, color } as DetectedStone);
    }
    return out;
  }, [refinedStones, hints]);

  // ── ③c 标记黑白空：点某一格 → 添加/切换 hint → 覆盖该格 ──
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

  // 手动覆盖结果：仅改被点的那一格
  const finalStones = finalizedStones;

  // ── ②b 计算死子：把最终棋盘 signMap 发给后端，打 Monte Carlo 领地概率图 ──
  const computeDeadStones = useCallback(async () => {
    if (result === null) return;
    setComputingDeadstones(true);
    setError(null);
    try {
      // 由当前最终棋子构造成 signMap[y][x]：1=黑 / -1=白 / 0=空
      const size = result.boardSize;
      const signMap: number[][] = Array.from({ length: size }, () => Array<number>(size).fill(0));
      for (const s of finalStones) {
        if (s.x < 0 || s.x >= size || s.y < 0 || s.y >= size) continue;
        signMap[s.y][s.x] = s.color === 'black' ? 1 : -1;
      }
      const body = JSON.stringify({ signMap, iterations: 2000 });
      const res = await fetch('/api/v1/deadstones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
      }
      const data: DeadStonesResponse = await res.json();
      setDeadstones(data);
    } catch (e) {
      setError(`计算死子失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setComputingDeadstones(false);
    }
  }, [result, finalStones]);

  // ── ③d 本地记分：死子已知后纯前端算，亚毫秒级零网络往返 ──
  // 死子集与领地（prob 图补单官）合成到棋盘 —— 死子 × + 领地半透明色叠画。
  const scoreResult = useMemo<
    | (ScoreBreakdown & { territoryMap: number[][]; deadStoneSet: Set<string>; winner: 'black' | 'white' | 'draw' })
    | null
  >(() => {
    if (result === null || deadstones === null) return null;
    const size = result.boardSize;
    const signMap: number[][] = Array.from({ length: size }, () => Array<number>(size).fill(0));
    for (const s of finalStones) {
      if (s.x < 0 || s.x >= size || s.y < 0 || s.y >= size) continue;
      signMap[s.y][s.x] = s.color === 'black' ? 1 : -1;
    }
    const deadStoneSet = deadStonesToSet(deadstones.deadStones);
    const est = computeScore(signMap, deadStoneSet, KOMI, deadstones.probabilityMap);

    const winner = est.blackScore > est.whiteScore ? 'black' : est.whiteScore > est.blackScore ? 'white' : 'draw';
    return {
      blackTerritory: est.blackTerritory,
      whiteTerritory: est.whiteTerritory,
      blackDeadStones: est.blackDeadStones,
      whiteDeadStones: est.whiteDeadStones,
      blackScore: est.blackScore,
      whiteScore: est.whiteScore,
      komi: KOMI,
      territoryMap: est.territoryMap,
      deadStoneSet,
      winner,
    };
  }, [result, deadstones, finalStones]);

  return (
    <div style={{ padding: 20, fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <h1>前后端分离棋盘识别样品</h1>
      <p style={{ color: '#555' }}>
        ① 上传棋局图片 → 后端识别 4 角 → 标注原图 → ② 上传推理 → ③ 本地拖阈值 / 标记黑白空
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
          <div style={{ marginTop: 8 }}>
            <button
              onClick={detectCorners}
              disabled={!imageUrl || detectingCorners}
              style={{ marginRight: 8 }}
            >
              {detectingCorners ? '识别角点中…' : '🤖 自动识别 4 角（后端）'}
            </button>
            {corners && (
              <button
                onClick={() => {
                  // 复位到默认内缩角点
                  const w = imgDims.width;
                  const h = imgDims.height;
                  const m = Math.min(w, h) * 0.05;
                  setCorners([
                    [m, m],
                    [w - 1 - m, m],
                    [w - 1 - m, h - 1 - m],
                    [m, h - 1 - m],
                  ] as BoardCorners);
                  setCornersManual(false);
                }}
              >
                重置角点
              </button>
            )}
          </div>
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
                {cornersManual ? '角点已标注' : '点击角点拖动'}
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
              <div style={{ marginTop: 10 }}>
                <button
                  onClick={computeDeadStones}
                  disabled={computingDeadstones}
                  style={{ padding: '6px 16px', fontWeight: 600 }}
                >
                  {computingDeadstones
                    ? '计算死子中…（Monte Carlo）'
                    : deadstones
                      ? '重新计算死子'
                      : '🀄 计算死子'}
                </button>
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
                territoryMap={scoreResult?.territoryMap ?? null}
                deadStoneSet={scoreResult?.deadStoneSet}
              />
              <div style={{ marginTop: 8, fontSize: 13 }}>
                <span>检出 {finalStones.length} 子 </span>
                {deadstones && (
                  <span style={{ marginLeft: 8, color: '#555' }}>
                    · 死子（黑 {deadstones.blackDeadStones} / 白 {deadstones.whiteDeadStones}）
                  </span>
                )}
              </div>
              {scoreResult && (
                <div
                  style={{
                    marginTop: 12,
                    borderTop: '1px solid #eee',
                    paddingTop: 10,
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr',
                    gap: 16,
                    alignItems: 'start',
                  }}
                >
                  <ScoreDisplay score={scoreResult} winner={scoreResult.winner} />
                  <div style={{ fontSize: 12, color: '#888', lineHeight: 1.6 }}>
                    领地按死子清零后 flood fill 封口判官子，单官点用后端概率图 ±0.2 阈值补齐
                    （<b>估计路径</b>）；分数 = 领地 + 敌死子（黑）+ komi（白）。
                    <br />
                    蓝色方块 = 黑地，橙色方块 = 白地，红 × = 死子。
                  </div>
                </div>
              )}
            </>
          ) : (
            <div style={{ color: '#999', border: '1px dashed #ccc', padding: 40, textAlign: 'center' }}>
              上传后在此显示对齐棋盘与微调控件
            </div>
          )}
          {error && <div style={{ color: '#c00', marginTop: 8 }}>⚠ {error}</div>}
        </section>
      </div>

      <p style={{ marginTop: 20, fontSize: 12, color: '#999' }}>
        后端 GET /health 检查模型；本样品前端假设服务已运行在 :8000（见 vite proxy）。
      </p>
    </div>
  );
}