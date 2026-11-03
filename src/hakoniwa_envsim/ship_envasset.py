# ship_envasset.py
"""
箱庭海上環境アセット (HakoShipEnv)
-----------------------------------
BVHベースのfastsearchを用いて、船舶位置（自船・他船）から
局所の海上環境パラメータ（風・潮流・水深・潮位・視程・波浪スペクトル）を検索し、
PDUに Disturbance として書き戻す。

依存モジュール:
  - hakopy
  - hakoniwa_pdu
  - fastsearch
  - asset.{ship_io, ship_manager, env_runtime}
"""

import sys
import time
import hakopy

from hakoniwa_envsim.asset.ship_manager import ShipManager
from hakoniwa_envsim.asset.ship_io import ShipIO
from hakoniwa_envsim.asset.env_runtime import EnvRuntime


# === globals ===
delta_time_usec = 0
config_path = ''
area_config_dir = ''
runtime: EnvRuntime = None
manager: ShipManager = None


# === 基本コールバック ===
def my_on_initialize(context):
    print("[ShipEnvAsset] Initialize event")
    return 0


def my_on_reset(context):
    print("[ShipEnvAsset] Reset event")
    return 0


def my_sleep():
    """箱庭シミュレータのクロックに同期してsleep"""
    global delta_time_usec
    if not hakopy.usleep(delta_time_usec):
        return False
    time.sleep(delta_time_usec / 1_000_000.0)
    return True


# === メイン処理 ===
def on_manual_timing_control(context):
    """
    手動タイミング制御ループ。
    自船および他船の位置を監視し、局所の海上環境プロパティから
    外乱(風/潮流/水深/波浪スペクトル/視程)をPDUに書き戻す。
    """
    global runtime, manager, config_path, area_config_dir
    print("[ShipEnvAsset] Start Maritime Environment Control")

    # --- 初期化 ---
    runtime = EnvRuntime.init(config_path, area_config_dir, depth=8, leaf_capacity=1)
    manager = ShipManager.from_config(config_path)

    print(f"[ShipEnvAsset] Environment loaded: areas={len(runtime.env.areas)} links={len(runtime.env.links)}")
    print(f"[ShipEnvAsset] Ships detected: {len(manager.ships)}")

    # --- メインループ ---
    while True:
        if not my_sleep():
            break

        runtime.pdu.run_nowait()

        # 各船舶ごとに処理
        for s in manager.ships:
            pose = s.read_pose(runtime.pdu)
            if pose is None:
                continue

            x, y, z = float(pose.linear.x), float(pose.linear.y), float(pose.linear.z)
            area_id, props = runtime.env.get_property_at(x, y, z)

            # プロパティ → 船舶用 Disturbance
            disturbance = ShipIO.make_disturbance(props)
            ok = s.write_disturbance(runtime.pdu, disturbance)
            if not ok:
                pass

    return 0


# === hakopy 登録 ===
my_callback = {
    'on_initialize': my_on_initialize,
    'on_simulation_step': None,
    'on_manual_timing_control': on_manual_timing_control,
    'on_reset': my_on_reset,
}


# === エントリポイント ===
def main():
    global delta_time_usec, config_path, area_config_dir

    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <pdu_config_path> <delta_time_msec> <area_config_dir>")
        return 1

    config_path = sys.argv[1]
    delta_time_usec = int(sys.argv[2]) * 1000
    area_config_dir = sys.argv[3]

    asset_name = 'HakoShipEnv'

    print(f"[ShipEnvAsset] Registering asset '{asset_name}'")
    ret = hakopy.asset_register(asset_name, config_path, my_callback,
                                delta_time_usec, hakopy.HAKO_ASSET_MODEL_PLANT)
    if not ret:
        print("[ERROR] Failed to register maritime env asset")
        return 1

    print("[ShipEnvAsset] Start simulation...")
    ret = hakopy.start()
    print(f"[ShipEnvAsset] DONE (status={ret})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
