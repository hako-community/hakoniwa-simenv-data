# asset/__init__.py
"""
Asset layer: hakoniwa PDU と fastsearch を橋渡しする薄いラッパ群。

- DroneIO / ShipIO: 1機/1隻ぶんの入出力（pos 読み、disturb/maritime_disturbance 書き）
- DroneManager / ShipManager: config(json) を解析して IO インスタンスを提供
- EnvRuntime: Environment と PduManager の初期化・保持
"""
from .drone_io import DroneIO
from .drone_manager import DroneManager
from .ship_io import ShipIO
from .ship_manager import ShipManager
from .env_runtime import EnvRuntime

__all__ = [
    "drone_io", "drone_manager", "ship_io", "ship_manager", "env_runtime",
    "DroneIO", "DroneManager", "ShipIO", "ShipManager", "EnvRuntime"
]
