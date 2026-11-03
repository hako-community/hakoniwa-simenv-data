# asset/ship_manager.py
"""
船舶マネージャ (ShipManager)
--------------------------
PDU定義ファイル (pdu_config_ship.json など) から
自船・他船の ShipIO インスタンスを検出し管理する。
"""

import json
from typing import List, Optional, Dict, Any, Set

from .ship_io import ShipIO, SHIP_POS_ORG, SHIP_DISTURB_ORG, FALLBACK_DISTURB_ORG


class ShipManager:
    """
    config(json) を解析して ShipIO を列挙。
    - robots[].name を PDU名として使用
    - shm_pdu_readers / shm_pdu_writers の org_name 整合性をチェック
    """

    def __init__(self, ships: List[ShipIO]):
        self.ships = ships

    @classmethod
    def from_config(cls, config_path: str) -> "ShipManager":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = json.load(f)

        ships: List[ShipIO] = []
        for rob in cfg.get("robots", []):
            name: Optional[str] = rob.get("name")
            if not name:
                continue

            reader_orgs: Set[str] = {r.get("org_name") for r in rob.get("shm_pdu_readers", []) if r.get("org_name")}
            writer_orgs: Set[str] = {w.get("org_name") for w in rob.get("shm_pdu_writers", []) if w.get("org_name")}

            # 位置読み出し org: 通常は 'pos'
            pos_org = SHIP_POS_ORG
            if SHIP_POS_ORG not in writer_orgs and SHIP_POS_ORG not in reader_orgs:
                # 警告ログ（ただし動作は継続）
                print(f"[INFO] robot '{name}' has no '{SHIP_POS_ORG}' org, will attempt default '{SHIP_POS_ORG}'")

            # 外乱書き込み org: 'maritime_disturbance' または 'disturb'
            disturb_org = SHIP_DISTURB_ORG
            if SHIP_DISTURB_ORG in reader_orgs or SHIP_DISTURB_ORG in writer_orgs:
                disturb_org = SHIP_DISTURB_ORG
            elif FALLBACK_DISTURB_ORG in reader_orgs or FALLBACK_DISTURB_ORG in writer_orgs:
                disturb_org = FALLBACK_DISTURB_ORG

            ships.append(ShipIO(name=name, pos_org=pos_org, disturb_org=disturb_org))

        return cls(ships)
