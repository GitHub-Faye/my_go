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
