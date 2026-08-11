"""/api/v1/corners 用 moku 模型的测试：合成棋盘 → 应有 4 角。"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import main

@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)

def _board(size_px=700, path=None):
    if path is None:
        path = Path("/tmp/moku_corner_synth.png")
    img = Image.new("RGB", (size_px, size_px), (210, 178, 140))
    d = ImageDraw.Draw(img)
    g = 19
    for i in range(g):
        x = y = 30 + i * (size_px - 60) // (g - 1)
        d.line([(x, 30), (x, size_px - 30)], fill=(60, 40, 20), width=2)
        d.line([(30, y), (size_px - 30, y)], fill=(60, 40, 20), width=2)
    img.save(path)
    return path

def test_corners_endpoint_moku(client, tmp_path):
    p = _board(path=tmp_path / "b.png")
    with p.open("rb") as f:
        res = client.post("/api/v1/corners", files={"image": ("b.png", f, "image/png")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["order"] == "TL/TR/BR/BL"
    assert body["cornersDetected"] is True
    corners = body["corners"]
    assert len(corners) == 4
    for x, y in corners:
        assert 0 <= x < 700 and 0 <= y < 700
    # 期望合成角点 (30,30)/(670,30)/(670,670)/(30,670)，允许 ±~60px 误差
    expected = [(30, 30), (670, 30), (670, 670), (30, 670)]
    for (x, y), (ex, ey) in zip(corners, expected):
        assert abs(x - ex) < 60, f"x {x} vs {ex}"
        assert abs(y - ey) < 60, f"y {y} vs {ey}"

def test_corners_fallback_when_degenerate(client, tmp_path):
    """moku 角点塌缩/退化时，/api/v1/corners 回退经典 CV，而非几何外推。

    构造一张饱和棋盘色、可被经典 CV 稳定检出的图片：渲染一块覆盖大片的
    木色饱和四边形（模拟棋盘），moku 即便识别出塌缩角，也应最终返回一个
    能覆盖大片内容、近似正方形的四角（源自经典 CV 轮廓），cornersDetected=false。
    """
    import cv2
    import numpy as np

    def _masked_wood(size=700) -> Image.Image:
        # 大片木色（饱和）棋盘 + 灰背景，确保饱和度掩码轮廓稳定
        board = np.full((size, size, 3), (150, 110, 60), np.uint8)
        m = np.zeros((size, size), np.uint8)
        cv2.rectangle(m, (60, 60), (size - 60, size - 60), 255, -1)
        bg = np.full((size, size, 3), (120, 120, 120), np.uint8)
        out = np.where(m[..., None].astype(bool), board, bg)
        return Image.fromarray(out)

    p = _masked_wood()
    path = tmp_path / "deg.png"
    p.save(path)
    with path.open("rb") as f:
        res = client.post("/api/v1/corners", files={"image": ("deg.png", f, "image/png")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["order"] == "TL/TR/BR/BL"
    # 木色棋盘可直接被经典 CV 检出，四个角应覆盖大片内容（近 60..640 内缩）
    assert len(body["corners"]) == 4
    corners = body["corners"]
    for x, y in corners:
        assert 0 <= x < 700 and 0 <= y < 700
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    # 覆盖大片：包围盒面积 > 图像 40%（Moku 塌缩内缩只有 ~5%）
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    assert bbox_area > 700 * 700 * 0.4, f"bbox {bbox_area} 太小，疑似塌缩/内缩未回退"


def test_corners_fallback_cv_with_moku_raw(client, tmp_path):
    """经典 CV 回退响应携带 mokuRawCorners，供前端对照 Moku 原始角点。"""
    import cv2
    import numpy as np

    board = np.full((400, 400, 3), (150, 110, 60), np.uint8)
    m = np.zeros((400, 400), np.uint8)
    cv2.rectangle(m, (30, 30), (370, 370), 255, -1)
    bg = np.full((400, 400, 3), (130, 130, 130), np.uint8)
    out = np.where(m[..., None].astype(bool), board, bg)
    p = tmp_path / "deg2.png"
    Image.fromarray(out).save(p)
    with p.open("rb") as f:
        res = client.post("/api/v1/corners", files={"image": ("deg2.png", f, "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert "mokuRawCorners" in body
    assert body["mokuRawCorners"] is not None
    assert len(body["mokuRawCorners"]) == 4


def test_corners_feeds_recognize(client, tmp_path):
    """经典 CV 回退得到的 corners 仍可直接喂给 /api/v1/recognize 完成识别。"""
    p = _board(path=tmp_path / "b2.png")
    with p.open("rb") as f:
        res = client.post("/api/v1/corners", files={"image": ("b2.png", f, "image/png")})
    assert res.status_code == 200
    corners = res.json()["corners"]
    flat = ",".join(f"{x},{y}" for x, y in corners)
    with p.open("rb") as f:
        res2 = client.post(
            f"/api/v1/recognize?corners={flat}",
            files={"image": ("b2.png", f, "image/png")},
            data={"boardSize": "19"},
        )
    assert res2.status_code == 200, res2.text
    body = res2.json()
    assert body["cornersDetected"] is True
    assert body["gridCorners"] is not None


def test_adjust_collapsed_top_corners_keeps_reliable_anchors():
    """顶边两角塌缩到图片最上方时，保留可靠底边(BL/BR)为锚，把 TL/TR 重新推导
    到覆盖大片内容的位置。这是用户报告的 moku 示例。"""
    from kaya_go.corners import adjust_degenerate_corners

    W, H = 1300, 1310
    corners = (
        (9.550153493881226, 74.93806225061417),
        (138.35674184560776, 128.2246584892273),
        (1286.2567241191864, 1305.0660982131958),
        (82.79735231399536, 1303.975350022316),
    )
    out, adjusted = adjust_degenerate_corners(corners, W, H)
    assert adjusted is True
    tl, tr, br, bl = out
    # 可靠底角(BR/BL)原样保留
    assert abs(br[0] - corners[2][0]) < 0.5 and abs(br[1] - corners[2][1]) < 0.5
    assert abs(bl[0] - corners[3][0]) < 0.5 and abs(bl[1] - corners[3][1]) < 0.5
    # 顶角被推离图片顶端、铺开到与底边同宽 → 覆盖大片
    assert tl[1] < 200 and tr[1] < 200          # 不再紧贴图片顶端
    assert tl[0] < 200 and tr[0] > W - 400       # 顶边铺开到右侧
    # 修复后 4 点构成覆盖大片的近似矩形：四条边都较长
    import math

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    top = dist(tl, tr)
    bottom = dist(br, bl)
    assert top > 1000 and bottom > 1000          # 上下边都宽


def test_adjust_skips_normal_quad():
    """正常展开、可覆盖大片的四边形不应被误判/修改。"""
    from kaya_go.corners import adjust_degenerate_corners

    corners = ((100, 100), (1100, 80), (1200, 1200), (50, 1150))
    out, adjusted = adjust_degenerate_corners(corners, 1300, 1310)
    assert adjusted is False
    assert out == corners


# ── mask 范围约束：CV 只在 moku 仿四边形内找点（用户设计第 ④ 步）──────────


def _masked_board_scope(W, H, scope_quad, board_pts):
    """构造饱和棋盘色(木色) + 灰背景的 RGBA 图，棋盘限定在 board_pts 多边形内。"""
    import cv2
    import numpy as np

    out = np.full((H, W, 4), (120, 120, 120, 255), np.uint8)  # 灰背景(低饱和)
    board = np.array([150, 110, 60, 255], np.uint8)  # 饱和木色棋盘
    cv2.fillPoly(
        out, [np.array(board_pts, np.int32)], board[:3].tolist()
    )
    return out


def test_find_board_corners_mask_restricts_scope():
    """mask_corners 把 CV 搜索范围限制在仿四边形内：范围外棋盘不被检出/干扰。"""
    from kaya_go.classic_cv import find_board_corners, _quad_mask, _board_mask
    import numpy as np

    W, H = 1356, 1420
    # 一张图：棋盘(饱和木色)实际占中间一块，但图里另有一块范围外的大色斑
    out = np.full((H, W, 4), (120, 120, 120, 255), np.uint8)
    board = (150, 110, 60, 255)
    import cv2

    # 棋盘真实位置：占据 (500,500)-(1100,1200) 附近
    cv2.fillPoly(out, [np.array([(500, 600), (1000, 500), (1050, 1150), (500, 1200)], np.int32)], board[:3])
    # 范围外的一个亮色块（噪声，不在 mask 里）
    cv2.fillPoly(out, [np.array([(10, 10), (300, 10), (300, 300), (10, 300)], np.int32)], (250, 120, 40))

    # 仿四边形 mask 精确框住棋盘（稍偏左上的搜索范围）
    scope = ((400, 500), (1100, 480), (1120, 1250), (420, 1280))
    # find_board_corners 直接返回有序四角
    corners = find_board_corners(out, mask_corners=scope)
    assert corners is not None
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    # 检出的角点应落在 scope 范围内（不含图像边缘 0/1356 或 0/1420）
    assert min(xs) >= 350 and max(xs) <= 1200
    assert min(ys) >= 400 and max(ys) <= 1350
    # 不应把左上角那范围外噪声色块当棋盘（范围外不框到它）
    assert max(xs) > 500 and max(ys) > 900   # 检出的是真实棋盘，不是左上角斑


def test_find_board_corners_mask_returns_none_outside():
    """搜索范围内没有棋盘(无饱和掩码)时，CV 在 mask 内检不出 → None。"""
    from kaya_go.classic_cv import find_board_corners
    import numpy as np

    W, H = 600, 600
    # 全灰背景（低饱和，无棋盘）
    out = np.full((H, W, 4), (120, 120, 120, 255), np.uint8)
    scope = ((100, 100), (500, 100), (500, 500), (100, 500))
    # mask 内也没有棋盘轮廓 → 返回 None（调用方应退回仿四边形）
    assert find_board_corners(out, mask_corners=scope) is None


def test_recover_uses_anchors_not_image_edges():
    """端到端：真实塌缩数据(moku TL/TR 挤到顶部)重建后保留可靠锚点，不被当整图。"""
    from kaya_go.corners import (
        quad_is_plausible,
        select_reliable_candidates,
        order_corners,
    )
    from kaya_go.moku_postprocess import _infer_top4
    from kaya_go.types import MokuRawDetection

    W, H = 1356, 1420
    moku = (
        (9.550153493881226, 74.93806225061417),
        (138.35674184560776, 128.2246584892273),
        (1286.2567241191864, 1305.0660982131958),
        (82.79735231399536, 1303.975350022316),
    )
    c = order_corners(list(moku))
    # 塌缩（TL/TR 挤到顶）→ 不构成近似四边形
    assert quad_is_plausible(c, W, H) is False

    # 可靠锚点应是构成最大近似三角形的 3 点（塌缩角 TR 被剔除）
    rel = select_reliable_candidates(list(c), W, H)
    assert len(rel) == 3
    # 塌缩角 TR(138,128) 不应进可靠锚点；右下 BR/BL 必须保留
    assert all(abs(p[0] - 138.36) > 20 or abs(p[1] - 128.22) > 20 for p in rel), f"塌缩角不该作锚点，得到 {rel}"
    assert any(abs(p[0] - 1286.26) < 1 and abs(p[1] - 1305.07) < 1 for p in rel)
    assert any(abs(p[0] - 82.80) < 1 and abs(p[1] - 1303.98) < 1 for p in rel)

    # 重建仿四边形：其中两个角保留自可靠锚点(右下)
    rebuilt = _infer_top4(
        [MokuRawDetection(cx=x, cy=y, class_id=2, score=0.0) for x, y in rel],
        W,
        H,
    )
    rebuilt = order_corners(rebuilt)
    assert quad_is_plausible(rebuilt, W, H) is True  # 重建后是近似四边形
    # 且四个角里至少有 2 个与原始可靠锚点重合（锚点被保留，未丢）
    coincident = 0
    for (rx, ry) in rebuilt:
        if any(abs(rx - mx) < 1.0 and abs(ry - my) < 1.0 for (mx, my) in rel):
            coincident += 1
    assert coincident >= 2, f"重建 quad{rebuilt} 应保留 ≥2 个可靠锚点"
    # 不应出现整盘内缩退化(零附近)或图像边缘四角
    for (rx, ry) in rebuilt:
        assert rx > 1 and ry > 1 and rx < W - 1 and ry < H - 1
