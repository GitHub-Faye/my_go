"""识别接口冒烟测试：用合成的 19 路棋盘图验证端到端流程。"""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from main import app
from kaya_go.detector import MokuDetector


def _make_board(size_px: int = 700, g: int = 19, path: Path | None = None) -> Path:
    if path is None:
        path = Path("/tmp/kaya_synth_board.png")
    img = Image.new("RGB", (size_px, size_px), (210, 178, 140))
    d = ImageDraw.Draw(img)
    for i in range(g):
        x = y = 30 + i * (size_px - 60) // (g - 1)
        d.line([(x, 30), (x, size_px - 30)], fill=(60, 40, 20), width=2)
        d.line([(30, y), (size_px - 30, y)], fill=(60, 40, 20), width=2)
    for xx, yy in [
        (30 + 3 * (size_px - 60) // 18, 30 + 3 * (size_px - 60) // 18),
        (30 + 15 * (size_px - 60) // 18, 30 + 4 * (size_px - 60) // 18),
    ]:
        d.ellipse([xx - 14, yy - 14, xx + 14, yy + 14], fill=(30, 30, 30))
    for xx, yy in [
        (30 + 8 * (size_px - 60) // 18, 30 + 8 * (size_px - 60) // 18),
        (30 + 10 * (size_px - 60) // 18, 30 + 11 * (size_px - 60) // 18),
    ]:
        d.ellipse([xx - 14, yy - 14, xx + 14, yy + 14], fill=(240, 240, 240))
    img.save(path)
    return path


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_recognize_endpoint(client, tmp_path):
    img_path = _make_board(path=tmp_path / "board.png")
    with img_path.open("rb") as f:
        res = client.post(
            "/api/v1/recognize",
            files={"image": ("board.png", f, "image/png")},
            data={"boardSize": "19"},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["boardSize"] == 19
    assert body["cornersDetected"] is True
    assert len(body["stones"]) >= 2
    assert body["sgf"].startswith("(;GM[1]FF[4]SZ[19]")
    # signMap 为 19×19 矩阵；黑/白子数 ≤ 画上去的 2/2（白子置信度低可能漏检）
    sign_map = body["signMap"]
    assert isinstance(sign_map, list) and len(sign_map) == 19
    assert all(isinstance(row, list) and len(row) == 19 for row in sign_map)
    assert all(v in (-1, 0, 1) for row in sign_map for v in row)
    black_count = sum(1 for row in sign_map for v in row if v == 1)
    white_count = sum(1 for row in sign_map for v in row if v == -1)
    assert black_count == 2  # 黑子稳定检出
    assert white_count <= 2 and black_count + white_count >= 2


def test_recognize_empty_file(client):
    res = client.post(
        "/api/v1/recognize",
        files={"image": ("x.png", b"", "image/png")},
    )
    assert res.status_code == 400


def test_detector_direct():
    """直接调用检测器（不经 HTTP）也能跑通，且结果稳定。"""
    from kaya_go.detector import DEFAULT_MODEL_PATH

    d = MokuDetector(DEFAULT_MODEL_PATH)
    arr = np.asarray(Image.open(_make_board()).convert("RGBA"))
    r = d.detect(arr, board_size=19)
    assert r.corners_detected is True
    assert len(r.stones) == 2


def test_build_sign_map():
    """signMap 由 DetectedStone 构造：signMap[y][x]=1黑/-1白/0空。"""
    from kaya_go.types import DetectedStone, RecognitionResult

    stones = [
        DetectedStone(x=3, y=4, color="black"),
        DetectedStone(x=6, y=2, color="white"),
    ]
    smap = RecognitionResult(
        board_size=9,
        stones=stones,
        corners=((0, 0), (0, 1), (1, 1), (1, 0)),
        corners_detected=True,
        sgf="",
    ).build_sign_map()
    assert len(smap) == 9 and all(len(row) == 9 for row in smap)
    assert smap[4][3] == 1  # (x=3, y=4) 黑
    assert smap[2][6] == -1  # (x=6, y=2) 白
    assert sum(1 for row in smap for v in row if v) == 2


def _simple_board():
    """一个 9 路简单局面：左上黑独立块、右下白独立块，互不交战。"""
    smap = [[0] * 9 for _ in range(9)]
    # 左上黑块（形成可死的孤棋，被空点包围）
    for y, x in [(1, 1), (1, 2), (2, 1)]:
        smap[y][x] = 1
    # 右下白块
    for y, x in [(6, 6), (6, 7), (7, 6)]:
        smap[y][x] = -1
    return smap


def test_deadstones_endpoint(client):
    """POST /api/v1/deadstones 返回 probabilityMap 与 deadStones。"""
    res = client.post(
        "/api/v1/deadstones",
        json={"signMap": _simple_board(), "iterations": 200, "seed": 123},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["boardSize"] == 9
    prob = body["probabilityMap"]
    assert isinstance(prob, list) and len(prob) == 9
    assert all(len(row) == 9 for row in prob)
    assert all(isinstance(v, (int, float)) and -1 <= v <= 1 for row in prob for v in row)
    assert isinstance(body["deadStones"], list)
    for d in body["deadStones"]:
        assert set(d) == {"x", "y"}
        assert isinstance(d["x"], int) and isinstance(d["y"], int)
    assert body["blackDeadStones"] >= 0
    assert body["whiteDeadStones"] >= 0


def test_deadstones_invalid(client):
    """非法 signMap 被拒绝。"""
    # 非正方形
    r = client.post("/api/v1/deadstones", json={"signMap": [[0, 0], [0]]})
    assert r.status_code == 400
    # 非法值
    r = client.post("/api/v1/deadstones", json={"signMap": [[2, 0], [0, 0]]})
    assert r.status_code == 400
    # 空
    r = client.post("/api/v1/deadstones", json={"signMap": []})
    assert r.status_code == 400


def test_deadstones_empty_board_allalive(client):
    """空棋盘的死子应为空，且无请求体校验错误。"""
    smap = [[0] * 9 for _ in range(9)]
    res = client.post("/api/v1/deadstones", json={"signMap": smap})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["boardSize"] == 9
    assert body["deadStones"] == []
    assert body["blackDeadStones"] == 0 and body["whiteDeadStones"] == 0
