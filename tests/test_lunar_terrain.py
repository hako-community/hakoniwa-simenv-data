"""
月面の地形タイル（examples/datasets/lunar/<name>/）の契約の試験。

1 つの格子から作った HField（MuJoCo）・16bit 高さ画像（Unreal 等）・grid.json が食い違わないこと、
岩の配置表が grid.json の範囲と地面に合っていること、岩の数がサイズ頻度モデルの期待値に合うことを見る。
DEM の取得は要らない（リポジトリに入れた生成物を検査する）。

Run: python -m unittest tests.test_lunar_terrain -v
"""

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "lunar_terrain"))
TILES = sorted(p for p in (REPO / "examples" / "datasets" / "lunar").iterdir() if (p / f"{p.name}.grid.json").exists())


def load_tile(tile: Path):
    name = tile.name
    grid = json.loads((tile / f"{name}.grid.json").read_text(encoding="utf-8"))
    raw = (tile / grid["mujoco"]["file"]).read_bytes()
    nr, nc = np.frombuffer(raw[:8], dtype=np.int32)
    hf = np.frombuffer(raw[8:], dtype=np.float32).reshape(nr, nc)
    return name, grid, hf


class LunarTileTest(unittest.TestCase):
    def test_has_tiles(self):
        self.assertTrue(TILES, "examples/datasets/lunar にタイルが無い")

    def test_hfield_matches_grid(self):
        for tile in TILES:
            name, grid, hf = load_tile(tile)
            g = grid["grid"]
            self.assertEqual(hf.shape, (g["n"], g["n"]), name)
            self.assertAlmostEqual(float(hf.min()), g["z_local_min_m"], places=3)
            self.assertAlmostEqual(float(hf.max()), g["z_local_max_m"], places=3)
            # 原点（格子の中央）の高さは 0
            self.assertAlmostEqual(float(hf[g["n"] // 2, g["n"] // 2]), 0.0, places=4)
            sx, sy, sz, _ = grid["mujoco"]["size"]
            self.assertAlmostEqual(sx, g["half_extent_m"], places=6)
            self.assertAlmostEqual(sz, g["z_local_max_m"] - g["z_local_min_m"], places=3)
            self.assertAlmostEqual(math.hypot(*grid["north_dir_local"]), 1.0, places=9)

    def test_height16_matches_hfield(self):
        from PIL import Image

        for tile in TILES:
            name, grid, hf = load_tile(tile)
            q = np.asarray(Image.open(tile / grid["unreal"]["file"]), dtype=np.float64)
            zmin, zmax = grid["grid"]["z_local_min_m"], grid["grid"]["z_local_max_m"]
            z16 = zmin + q / 65535.0 * (zmax - zmin)
            # 画像は上の行 = +y、HField は行 0 = −y
            err = np.abs(np.flipud(z16) - hf).max()
            self.assertLess(err, (zmax - zmin) / 65535.0 * 0.5 + 1e-4, f"{name}: 16bit と HField の差 {err:.5f} m")

    def test_rocks_table(self):
        for tile in TILES:
            name, grid, hf = load_tile(tile)
            path = tile / f"{name}.rocks.json"
            if not path.exists():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            rocks = np.array(doc["rocks"], dtype=np.float64)
            self.assertEqual(len(rocks), doc["count"])
            model = doc["model"]
            half = grid["grid"]["half_extent_m"] - model["margin_m"]
            self.assertTrue((np.abs(rocks[:, :2]) <= half + 1.0).all(), "岩がタイルの範囲の外にある")
            self.assertTrue((np.hypot(rocks[:, 0], rocks[:, 1]) > model["clear_radius_m"] - 1.0).all(), "出発点の周りに岩がある")
            self.assertTrue((rocks[:, 3:6] > 0).all())
            self.assertTrue(np.allclose(np.linalg.norm(rocks[:, 6:10], axis=1), 1.0, atol=1e-5))
            self.assertTrue((rocks[:, 10] < len(doc["meshes"])).all())
            for mesh in doc["meshes"]:
                r = np.linalg.norm(np.reshape(mesh["vertices"], (-1, 3)), axis=1)
                self.assertLessEqual(r.max(), 1.0 + 1e-5, "絵のメッシュが楕円体の外に出ている")
                self.assertGreater(r.min(), 0.8)
                self.assertEqual(len(mesh["triangles"]) % 3, 0)

    def test_rocks_sit_on_ground(self):
        """岩の中心が地面から自分の高さ（c）の範囲にある：地面は HField を双線形でなく三角形で引くので MuJoCo のレイで"""
        import mujoco

        for tile in TILES:
            name, grid, hf = load_tile(tile)
            path = tile / f"{name}.rocks.json"
            if not path.exists():
                continue
            rocks = np.array(json.loads(path.read_text(encoding="utf-8"))["rocks"], dtype=np.float64)
            mj = grid["mujoco"]
            sx, sy, sz, base = mj["size"]
            m = mujoco.MjModel.from_xml_string(
                f'<mujoco><asset><hfield name="dem" file="{(tile / mj["file"]).as_posix()}"'
                f' size="{sx:.3f} {sy:.3f} {sz:.4f} {base:.3f}"/></asset>'
                f'<worldbody><geom type="hfield" hfield="dem" pos="0 0 {mj["geom_pos_z"]:.4f}"/></worldbody></mujoco>')
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            gid = np.zeros(1, dtype=np.int32)
            top = grid["grid"]["z_local_max_m"] + 50.0
            for r in rocks[:: max(1, len(rocks) // 300)]:
                dist = mujoco.mj_ray(m, d, np.array([r[0], r[1], top]), np.array([0, 0, -1.0]), None, 1, -1, gid)
                g = top - dist
                self.assertLess(abs(r[2] - g), r[5] + 0.05, f"{name}: 岩 ({r[0]:.1f}, {r[1]:.1f}) の中心が地面から離れている")

    def test_rock_count_follows_model(self):
        import rock_field

        for tile in TILES:
            name, grid, _ = load_tile(tile)
            path = tile / f"{name}.rocks.json"
            if not path.exists():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            mo = doc["model"]
            _, xs, cum = rock_field.golombek_cumulative(mo["k"], np.array([mo["dmin_m"]]))
            per_m2 = float(np.interp(mo["dmin_m"], xs, cum) - np.interp(mo["dmax_m"], xs, cum))
            half = grid["grid"]["half_extent_m"] - mo["margin_m"]
            expect = per_m2 * ((2 * half) ** 2 - math.pi * mo["clear_radius_m"] ** 2)
            # ポアソンの揺らぎ（標準偏差 √N）の 4 倍まで
            self.assertLess(abs(doc["count"] - expect), 4.0 * math.sqrt(expect), f"{name}: 岩 {doc['count']} 個・期待値 {expect:.0f}")

    def test_golombek_rapp(self):
        """k = 1% のとき、直径 0.2 m 以上の岩は約 0.007 個/m²（Golombek & Rapp 1997 の式を数値積分した値）"""
        import rock_field

        n, _, _ = rock_field.golombek_cumulative(0.01, np.array([0.2, 0.3, 0.5]))
        self.assertAlmostEqual(n[0], 0.00710, delta=0.0001)
        self.assertTrue(n[0] > n[1] > n[2] > 0)


if __name__ == "__main__":
    unittest.main()
