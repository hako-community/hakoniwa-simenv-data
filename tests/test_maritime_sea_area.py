"""test_maritime_sea_area.py - 海域プリセット幾何・潮流計算の単体テスト (hakoniwa-simenv-data)。"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_DIR / "examples" / "maritime"
sys.path.insert(0, str(EXAMPLES_DIR))

from sea_area import SeaArea


def test_uraga_suido_sea_area():
    """東京湾 浦賀水道プリセットの幾何と潮流の妥当性検証。"""
    json_path = EXAMPLES_DIR / "sea_areas" / "uraga_suido.json"
    assert json_path.exists(), f"Uraga suido json not found: {json_path}"

    area = SeaArea(str(json_path))
    assert area.cfg["name"] == "uraga_suido"
    assert "浦賀水道航路" in area.routes

    route = area.routes["浦賀水道航路"]
    # 航路中心線長はおよそ 14.8 km (14000〜16000 m)
    length = route.length()
    assert 14000.0 <= length <= 16000.0, f"Route length {length} m out of expected range"

    # 航路幅 (約 700〜1500 m)
    widths = route.widths()
    assert min(widths) >= 600.0
    assert max(widths) <= 2000.0

    # 潮流計算（t=0 での原点近傍の潮流ベクトル）
    cur_e, cur_n = area.current(0.0, 0.0, 0.0)
    assert isinstance(cur_e, float) and isinstance(cur_n, float)


def test_bisan_seto_sea_area():
    """瀬戸内海 備讃瀬戸プリセットの幾何と潮流の妥当性検証。"""
    json_path = EXAMPLES_DIR / "sea_areas" / "bisan_seto.json"
    assert json_path.exists(), f"Bisan seto json not found: {json_path}"

    area = SeaArea(str(json_path))
    assert area.cfg["name"] == "bisan_seto"
    assert len(area.routes) >= 1

    # 備讃瀬戸は潮流が強い海域（ピーク 2 kt 以上）
    c = area.cfg.get("current")
    assert c is not None
    assert c["peak_kt"] >= 2.0
