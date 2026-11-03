# tests/test_ship_io.py
"""
test_ship_io.py — 船舶用 Disturbance PDU 変換と海上環境スキーマの単体テスト
"""

import sys
import os
import math
import json
import pytest

# hakoniwa-simenv-data/src を path に追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hakoniwa_envsim.asset.ship_io import ShipIO
from hakoniwa_envsim.asset.ship_manager import ShipManager


def test_ship_io_defaults():
    """props が None の場合の安全なデフォルト値テスト"""
    d = ShipIO.make_disturbance(None)
    assert d.d_wind.value.x == 0.0
    assert d.d_wind.value.y == 0.0
    assert d.d_wind.value.z == 0.0
    assert d.d_temp.value == 15.0
    assert d.d_atm.sea_level_atm == 1.0

    # customs
    assert len(d.d_user_custom) == 3
    # c0: [gps_strength, visibility_m, rain_rate_mm_h]
    assert d.d_user_custom[0].data[0] == 1.0
    assert d.d_user_custom[0].data[1] == 10000.0
    assert d.d_user_custom[0].data[2] == 0.0

    # c1: [current_vx, current_vy, current_vz, water_density, depth_m, tide_z]
    assert d.d_user_custom[1].data[0] == 0.0
    assert d.d_user_custom[1].data[3] == 1025.0
    assert d.d_user_custom[1].data[4] == 50.0

    # c2: [wave_hs_m, wave_tp_s, wave_dir_rad, wave_gamma, wave_seed]
    assert d.d_user_custom[2].data[0] == 0.0
    assert d.d_user_custom[2].data[3] == 3.3


def test_ship_io_maritime_properties_roundtrip():
    """海上環境プロパティを設定した場合のエンコード・デコード（相互変換）テスト"""
    props = {
        "wind_velocity": [8.5, -2.0, 0.0],
        "temperature": 18.5,
        "sea_level_atm": 1.013,
        "gps_strength": 0.95,
        "visibility_m": 250.0,       # 濃霧
        "rain_rate_mm_h": 15.0,      # やや強い雨
        "current_velocity": [1.2, 0.3, 0.0], # 潮流 ~2.5 kt
        "water_density": 1026.5,
        "water_depth": 14.5,         # 浅水 (749GT船喫水 3.8m に対して H/d ~ 3.8)
        "tide_level": 0.8,           # 高潮位
        "wave_spectrum": {
            "hs_m": 2.2,
            "tp_s": 7.5,
            "wave_dir_deg": 45.0,     # 北東からの波
            "gamma": 3.3,
            "seed": 98765
        }
    }

    d = ShipIO.make_disturbance(props)

    # 1. 直接 Disturbance フィールド検証
    assert d.d_wind.value.x == pytest.approx(8.5)
    assert d.d_wind.value.y == pytest.approx(-2.0)
    assert d.d_temp.value == pytest.approx(18.5)

    # c0
    assert d.d_user_custom[0].data[0] == pytest.approx(0.95)
    assert d.d_user_custom[0].data[1] == pytest.approx(250.0)
    assert d.d_user_custom[0].data[2] == pytest.approx(15.0)

    # c1
    assert d.d_user_custom[1].data[0] == pytest.approx(1.2)
    assert d.d_user_custom[1].data[1] == pytest.approx(0.3)
    assert d.d_user_custom[1].data[3] == pytest.approx(1026.5)
    assert d.d_user_custom[1].data[4] == pytest.approx(14.5)
    assert d.d_user_custom[1].data[5] == pytest.approx(0.8)

    # c2
    assert d.d_user_custom[2].data[0] == pytest.approx(2.2)
    assert d.d_user_custom[2].data[1] == pytest.approx(7.5)
    assert d.d_user_custom[2].data[2] == pytest.approx(math.radians(45.0))
    assert d.d_user_custom[2].data[3] == pytest.approx(3.3)
    assert int(d.d_user_custom[2].data[4]) == 98765

    # 2. 辞書逆展開検証
    recovered = ShipIO.parse_disturbance_to_dict(d)
    assert recovered["wind_velocity"] == [pytest.approx(8.5), pytest.approx(-2.0), pytest.approx(0.0)]
    assert recovered["visibility_m"] == pytest.approx(250.0)
    assert recovered["rain_rate_mm_h"] == pytest.approx(15.0)
    assert recovered["current_velocity"][0] == pytest.approx(1.2)
    assert recovered["water_depth"] == pytest.approx(14.5)
    assert recovered["wave_spectrum"]["hs_m"] == pytest.approx(2.2)
    assert recovered["wave_spectrum"]["seed"] == 98765


def test_ship_manager_from_config():
    """pdu_config_ship.json からの ShipIO 列挙テスト"""
    candidates = [
        r"D:\work_ship\shipnavi\pdu_config_ship.json",
        r"D:\work_ship\pdu_config_ship.json"
    ]
    config_path = next((c for c in candidates if os.path.exists(c)), None)
    if not config_path:
        pytest.skip("pdu_config_ship.json not found in shipnavi or work_ship")

    mgr = ShipManager.from_config(config_path)
    ship_names = [s.name for s in mgr.ships]
    assert "OwnShip" in ship_names
    assert "TargetShip01" in ship_names
    assert "TargetShipFishing" in ship_names

    own_ship = next(s for s in mgr.ships if s.name == "OwnShip")
    assert own_ship.pos_org == "pos"
    assert own_ship.disturb_org == "maritime_disturbance"


def test_maritime_schema_validation():
    """新スキーマ area_properties.schema.json によるバリデーションテスト"""
    import jsonschema

    schema_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "src", "hakoniwa_envsim", "schemas", "area_properties.schema.json"
    ))
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    valid_maritime_data = {
        "area_properties": [
            {
                "id": "prop_tokyo_bay_uraga",
                "properties": {
                    "wind_velocity": [5.0, 2.0, 0.0],
                    "temperature": 20.0,
                    "sea_level_atm": 1.0,
                    "gps_strength": 1.0,
                    "visibility_m": 8000.0,
                    "rain_rate_mm_h": 0.0,
                    "current_velocity": [0.8, -0.2, 0.0],
                    "water_density": 1025.0,
                    "water_depth": 35.0,
                    "tide_level": 0.2,
                    "wave_spectrum": {
                        "hs_m": 1.5,
                        "tp_s": 5.8,
                        "wave_dir_deg": 180.0,
                        "gamma": 3.3,
                        "seed": 1001
                    }
                }
            }
        ]
    }

    # 例外が出なければ成功
    jsonschema.validate(instance=valid_maritime_data, schema=schema)
