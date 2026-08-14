"""静态局势分析接口（/api/v1/analyze-position）冒烟测试。

- payload 用 GET query 的 JSON 字符串（对齐端点实现）。
- 模型文件缺省时用 TestClient 调真实 ONNX 会话，故仅做轻量冒烟，
  其余信号已由 `test_static_analysis` 的单测路径覆盖。
"""
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from main import app

    return TestClient(app)


def _empty(size: int = 19) -> str:
    return json.dumps([[0] * size for _ in range(size)])


def test_analyze_empty_board_black_first(client):
    r = client.get(
        "/api/v1/analyze-position",
        params={"signMap": _empty(), "nextToPlay": "B", "includeOwnership": "false"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["winRate"] <= 1
    assert isinstance(body["scoreLead"], (int, float))
    assert body["currentTurn"] == "B"
    assert body["visits"] == 1
    assert body["moveSuggestions"]  # 至少有一个候选
    assert body["ownership"] is None
    assert "model" in body


def test_analyze_omits_next_to_play_infers_black(client):
    # 空盘子数相同 → 推断为 B（黑先）
    r = client.get(
        "/api/v1/analyze-position",
        params={"signMap": _empty(), "includeOwnership": "false"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["currentTurn"] == "B"


def test_analyze_ambiguous_requires_explicit(client):
    # 子数差 >1：无 nextToPlay → 503；显式传入 → 200
    m = [[0] * 9 for _ in range(9)]
    m[0][0] = 1
    m[0][1] = 1
    m[0][2] = -1
    r = client.get("/api/v1/analyze-position", params={"signMap": json.dumps(m)})
    assert r.status_code == 503
    r2 = client.get(
        "/api/v1/analyze-position",
        params={"signMap": json.dumps(m), "nextToPlay": "B", "includeOwnership": "false"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["currentTurn"] == "B"


def test_analyze_bad_signmap_400(client):
    r = client.get(
        "/api/v1/analyze-position",
        params={"signMap": "not-json"},
    )
    assert r.status_code == 400
