/**
 * 记分明细面板 —— 展示死子、领地、最终分数。
 *
 * 输入已由上层计算好的 score（computeScore 产物）与 territoryMap，
 * 本组件纯展示、无逻辑，便于复用/替换样式。
 */

export interface ScoreBreakdown {
  blackTerritory: number;
  whiteTerritory: number;
  blackDeadStones: number;
  whiteDeadStones: number;
  blackScore: number;
  whiteScore: number;
  komi: number;
}

interface ScoreDisplayProps {
  score: ScoreBreakdown;
  /** 胜方：'black' | 'white' | 'draw'，胜者绿色加粗。 */
  winner: 'black' | 'white' | 'draw';
}

function Row({ label, black, white }: { label: string; black: number; white: number }) {
  return (
    <tr>
      <td style={{ padding: '2px 12px 2px 0', color: '#333' }}>{label}</td>
      <td style={{ padding: '2px 12px', textAlign: 'right', fontWeight: 600 }}>{black}</td>
      <td style={{ padding: '2px 0', textAlign: 'right', fontWeight: 600 }}>{white}</td>
    </tr>
  );
}

export default function ScoreDisplay({ score, winner }: ScoreDisplayProps) {
  return (
    <div style={{ fontSize: 14 }}>
      <table style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left', padding: '2px 12px 6px 0', borderBottom: '2px solid #ccc' }} />
            <th style={{ textAlign: 'right', padding: '2px 12px 6px', borderBottom: '2px solid #333' }}>黑 ●</th>
            <th style={{ textAlign: 'right', padding: '2px 0 6px', borderBottom: '2px solid #999' }}>白 ○</th>
          </tr>
        </thead>
        <tbody>
          <Row label="领地" black={score.blackTerritory} white={score.whiteTerritory} />
          <Row label="提子(死子)" black={score.blackDeadStones} white={score.whiteDeadStones} />
          <Row label="贴目 komi" black={0} white={score.komi} />
          <tr>
            <td style={{ padding: '6px 12px 2px 0', borderTop: '2px solid #ccc' }}>
              <strong>得分</strong>
            </td>
            <td style={{ padding: '6px 12px 2px', textAlign: 'right', borderTop: '2px solid #ccc' }}>
              <strong style={{ color: winner === 'black' ? '#0a0' : '#111' }}>{score.blackScore}</strong>
            </td>
            <td style={{ padding: '6px 0 2px', textAlign: 'right', borderTop: '2px solid #ccc' }}>
              <strong style={{ color: winner === 'white' ? '#0a0' : '#111' }}>{score.whiteScore}</strong>
            </td>
          </tr>
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 13, color: '#555' }}>
        {winner === 'draw' ? (
          '和棋'
        ) : (
          <>
            <b style={{ color: winner === 'black' ? '#000' : '#888' }}>黑方</b> {winner === 'black' ? '胜' : '负'} ·{' '}
            <b style={{ color: winner === 'white' ? '#000' : '#888' }}>白方</b> {winner === 'white' ? '胜' : '负'}
          </>
        )}
      </div>
    </div>
  );
}
