# asset/ship_io.py
"""
船舶向け海上外乱 I/O モジュール (ShipIO)
----------------------------------------
箱庭の hako_msgs/Disturbance PDU を通じて、
空間海上環境（海上風・潮流・水深・潮位・視程・降雨・波浪スペクトル）を送受信する。

PDUレイアウト規約 (検討書 §3.3 (2) 準拠):
  d_wind.value.{x,y,z} : 海上風ベクトル [m/s] (ENU座標系)
  d_temp.value         : 気温 [degC]
  d_atm.sea_level_atm  : 海面気圧 [atm]
  d_user_custom[0].data: [gps_strength (0..1), visibility_m [m], rain_rate_mm_h [mm/h]]
  d_user_custom[1].data: [current_vx, current_vy, current_vz [m/s], water_density [kg/m^3], depth_m [m], tide_z [m]]
  d_user_custom[2].data: [wave_hs_m [m], wave_tp_s [s], wave_dir_rad [rad], wave_gamma, wave_seed]
"""

from dataclasses import dataclass
import math
from typing import Optional, Dict, Any

from hakoniwa_pdu.pdu_manager import PduManager

from hakoniwa_pdu.pdu_msgs.geometry_msgs.pdu_pytype_Twist import Twist
from hakoniwa_pdu.pdu_msgs.geometry_msgs.pdu_conv_Twist import pdu_to_py_Twist
from hakoniwa_pdu.pdu_msgs.hako_msgs.pdu_pytype_Disturbance import Disturbance
from hakoniwa_pdu.pdu_msgs.hako_msgs.pdu_conv_Disturbance import py_to_pdu_Disturbance
from hakoniwa_pdu.pdu_msgs.hako_msgs.pdu_pytype_DisturbanceUserCustom import DisturbanceUserCustom

SHIP_POS_ORG = "pos"
SHIP_DISTURB_ORG = "maritime_disturbance"
FALLBACK_DISTURB_ORG = "disturb"


@dataclass
class ShipIO:
    """
    1隻ぶんの船舶環境 I/O 担当。
    - name: PDU上のロボット名（例: 'OwnShip', 'TargetShip01'）
    - pos_org: 位置読取の org_name（デフォルト 'pos'）
    - disturb_org: 外乱出力の org_name（デフォルト 'maritime_disturbance'）
    """
    name: str
    pos_org: str = SHIP_POS_ORG
    disturb_org: str = SHIP_DISTURB_ORG

    # --- READ ---

    def read_pose(self, pdu_manager: PduManager) -> Optional[Twist]:
        """自船・他船の現在位置・姿勢(Twist)を読み取る。未準備なら None。"""
        raw = pdu_manager.read_pdu_raw_data(self.name, self.pos_org)
        if not raw:
            return None
        try:
            return pdu_to_py_Twist(raw)
        except Exception:
            return None

    # --- WRITE ---

    def write_disturbance(self, pdu_manager: PduManager, disturbance: Disturbance) -> bool:
        """
        Disturbance PDU を共有メモリへ書き込む。
        maritime_disturbance が未定義の場合は fallback として disturb への書き込みを試行。
        """
        try:
            raw = py_to_pdu_Disturbance(disturbance)
        except Exception as e:
            print(f"[ShipIO] Failed to convert Disturbance to PDU: {e}")
            return False

        for target_org in [self.disturb_org, FALLBACK_DISTURB_ORG]:
            try:
                ok = pdu_manager.flush_pdu_raw_data_nowait(self.name, target_org, raw)
                if ok:
                    return True
            except Exception:
                pass

        print(f"[WARN] ShipIO: Failed to flush disturbance: name={self.name} org={self.disturb_org}")
        return False

    # --- Utility ---

    @staticmethod
    def make_disturbance(props: Optional[Dict[str, Any]]) -> Disturbance:
        """
        area_properties(dict) → 船舶用 Disturbance PDU 変換。
        props が None または未設定の項目は安全な標準値（静穏海面、晴天、微風）で初期化。
        """
        d = Disturbance()
        d.d_wind.value.x = 0.0
        d.d_wind.value.y = 0.0
        d.d_wind.value.z = 0.0
        d.d_temp.value = 15.0        # 標準海面気温 15.0 degC
        d.d_atm.sea_level_atm = 1.0  # 標準気圧 1.0 atm

        # デフォルト値
        gps_strength = 1.0
        visibility_m = 10000.0       # 快晴視程 10 km
        rain_rate_mm_h = 0.0         # 降雨なし

        current_vx = 0.0
        current_vy = 0.0
        current_vz = 0.0
        water_density = 1025.0       # 標準海水密度 1025 kg/m^3
        water_depth = 50.0           # 十分な水深 50 m
        tide_z = 0.0

        wave_hs = 0.0                # 静水
        wave_tp = 6.0                # 標準うねり周期 6.0 s
        wave_dir_rad = 0.0
        wave_gamma = 3.3             # JONSWAP 標準
        wave_seed = 12345

        if props:
            # 1. 海上風
            wind = props.get("wind_velocity")
            if wind and len(wind) >= 3:
                d.d_wind.value.x = float(wind[0])
                d.d_wind.value.y = float(wind[1])
                d.d_wind.value.z = float(wind[2])

            # 2. 気温・気圧
            if "temperature" in props and props["temperature"] is not None:
                d.d_temp.value = float(props["temperature"])
            if "sea_level_atm" in props and props["sea_level_atm"] is not None:
                d.d_atm.sea_level_atm = float(props["sea_level_atm"])

            # 3. GPS・視程・降雨
            if "gps_strength" in props and props["gps_strength"] is not None:
                gps_strength = float(props["gps_strength"])
            if "visibility_m" in props and props["visibility_m"] is not None:
                visibility_m = float(props["visibility_m"])
            if "rain_rate_mm_h" in props and props["rain_rate_mm_h"] is not None:
                rain_rate_mm_h = float(props["rain_rate_mm_h"])

            # 4. 潮流・海水・水深・潮位
            curr = props.get("current_velocity")
            if curr and len(curr) >= 3:
                current_vx = float(curr[0])
                current_vy = float(curr[1])
                current_vz = float(curr[2])
            if "water_density" in props and props["water_density"] is not None:
                water_density = float(props["water_density"])
            if "water_depth" in props and props["water_depth"] is not None:
                water_depth = float(props["water_depth"])
            if "tide_level" in props and props["tide_level"] is not None:
                tide_z = float(props["tide_level"])

            # 5. 波浪スペクトル (JONSWAP)
            wave = props.get("wave_spectrum")
            if wave and isinstance(wave, dict):
                if "hs_m" in wave and wave["hs_m"] is not None:
                    wave_hs = float(wave["hs_m"])
                if "tp_s" in wave and wave["tp_s"] is not None:
                    wave_tp = float(wave["tp_s"])
                if "wave_dir_rad" in wave and wave["wave_dir_rad"] is not None:
                    wave_dir_rad = float(wave["wave_dir_rad"])
                elif "wave_dir_deg" in wave and wave["wave_dir_deg"] is not None:
                    wave_dir_rad = math.radians(float(wave["wave_dir_deg"]))
                if "gamma" in wave and wave["gamma"] is not None:
                    wave_gamma = float(wave["gamma"])
                if "seed" in wave and wave["seed"] is not None:
                    wave_seed = int(wave["seed"])

        # d_user_custom の構築
        c0 = DisturbanceUserCustom()
        c0.data = [float(gps_strength), float(visibility_m), float(rain_rate_mm_h)]

        c1 = DisturbanceUserCustom()
        c1.data = [float(current_vx), float(current_vy), float(current_vz),
                   float(water_density), float(water_depth), float(tide_z)]

        c2 = DisturbanceUserCustom()
        c2.data = [float(wave_hs), float(wave_tp), float(wave_dir_rad),
                   float(wave_gamma), float(wave_seed)]

        d.d_user_custom = [c0, c1, c2]
        return d

    @staticmethod
    def parse_disturbance_to_dict(d: Disturbance) -> Dict[str, Any]:
        """
        Disturbance PDU → 船舶環境パラメータ辞書への逆展開ユーティリティ。
        物理エンジン、Viewer、テスト検証で使用可能。
        """
        res: Dict[str, Any] = {
            "wind_velocity": [d.d_wind.value.x, d.d_wind.value.y, d.d_wind.value.z],
            "temperature": d.d_temp.value,
            "sea_level_atm": d.d_atm.sea_level_atm,
            "gps_strength": 1.0,
            "visibility_m": 10000.0,
            "rain_rate_mm_h": 0.0,
            "current_velocity": [0.0, 0.0, 0.0],
            "water_density": 1025.0,
            "water_depth": 50.0,
            "tide_level": 0.0,
            "wave_spectrum": {
                "hs_m": 0.0,
                "tp_s": 6.0,
                "wave_dir_rad": 0.0,
                "gamma": 3.3,
                "seed": 12345
            }
        }

        if hasattr(d, "d_user_custom") and d.d_user_custom:
            customs = d.d_user_custom
            # c0: [gps_strength, visibility_m, rain_rate_mm_h]
            if len(customs) > 0 and hasattr(customs[0], "data") and len(customs[0].data) >= 3:
                res["gps_strength"] = float(customs[0].data[0])
                res["visibility_m"] = float(customs[0].data[1])
                res["rain_rate_mm_h"] = float(customs[0].data[2])

            # c1: [current_vx, current_vy, current_vz, water_density, depth_m, tide_z]
            if len(customs) > 1 and hasattr(customs[1], "data") and len(customs[1].data) >= 6:
                res["current_velocity"] = [
                    float(customs[1].data[0]),
                    float(customs[1].data[1]),
                    float(customs[1].data[2])
                ]
                res["water_density"] = float(customs[1].data[3])
                res["water_depth"] = float(customs[1].data[4])
                res["tide_level"] = float(customs[1].data[5])

            # c2: [wave_hs_m, wave_tp_s, wave_dir_rad, wave_gamma, wave_seed]
            if len(customs) > 2 and hasattr(customs[2], "data") and len(customs[2].data) >= 5:
                res["wave_spectrum"]["hs_m"] = float(customs[2].data[0])
                res["wave_spectrum"]["tp_s"] = float(customs[2].data[1])
                res["wave_spectrum"]["wave_dir_rad"] = float(customs[2].data[2])
                res["wave_spectrum"]["gamma"] = float(customs[2].data[3])
                res["wave_spectrum"]["seed"] = int(customs[2].data[4])

        return res
