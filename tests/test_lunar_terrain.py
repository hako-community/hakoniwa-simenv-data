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
                self.assertGreaterEqual(r.min(), 0.8 - 1e-5, "絵のメッシュの凹凸が深すぎる（bump 0.2 まで）")
                self.assertEqual(len(mesh["triangles"]) % 3, 0)

    def test_rock_meshes_are_closed(self):
        """絵のメッシュは閉じた面：位置で見た辺はどれもちょうど 2 枚の三角形が共有し、面積 0 の三角形が無い
        （角を出すために頂点を分けても、つながりは変わらない）"""
        for tile in TILES:
            name, _, _ = load_tile(tile)
            path = tile / f"{name}.rocks.json"
            if not path.exists():
                continue
            for k, mesh in enumerate(json.loads(path.read_text(encoding="utf-8"))["meshes"]):
                p = np.reshape(mesh["vertices"], (-1, 3))
                t = np.reshape(mesh["triangles"], (-1, 3))
                area = np.linalg.norm(np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), axis=1)
                self.assertGreater(area.min(), 1e-8, f"mesh {k}: 面積 0 の三角形")
                key = [tuple(np.round(x, 4)) for x in p]
                edges = {}
                for tri in t:
                    for a, b in ((0, 1), (1, 2), (2, 0)):
                        e = tuple(sorted((key[tri[a]], key[tri[b]])))
                        edges[e] = edges.get(e, 0) + 1
                self.assertEqual(set(edges.values()), {2}, f"mesh {k}: 閉じていない（2 枚で共有されない辺がある）")

    def test_rocks_sit_on_ground(self):
        """岩の中心が地面から自分の高さ（c）の範囲にある：地面は HField を双線形でなく三角形で引くので MuJoCo のレイで"""
        import mujoco

        for tile in TILES:
            name, grid, hf = load_tile(tile)
            path = tile / f"{name}.rocks.json"
            if not path.exists():
                continue
            rocks = np.array(json.loads(path.read_text(encoding="utf-8"))["rocks"], dtype=np.float64)
            if (tile / f"{name}.craters.json").exists():
                # 小クレータのあるタイル：地面は細かい格子（crater_field.py）
                import crater_field
                meta, fine = crater_field.build_fine(tile)
                fx, fy = meta["half_extent_m"]
                for r in rocks[:: max(1, len(rocks) // 300)]:
                    g = float(crater_field.tri_height(fine, fx, fy, np.array(r[0]), np.array(r[1])))
                    self.assertLess(abs(r[2] - g), r[5] + 0.05, f"{name}: 岩 ({r[0]:.1f}, {r[1]:.1f}) の中心が地面から離れている")
                continue
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

    def test_tri_height_matches_mujoco(self):
        """crater_field.tri_height（numpy の三角形の補間）が MuJoCo の HField のレイと同じ高さ（細かい格子の土台）"""
        import mujoco
        import crater_field

        for tile in TILES:
            name, grid, _ = load_tile(tile)
            _, z = crater_field.load_coarse(tile)
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
            pts = np.random.default_rng(3).uniform(-sx * 0.99, sx * 0.99, (500, 2))
            ray = np.array([top - mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0, 0, -1.0]), None, 1, -1, gid) for x, y in pts])
            err = np.abs(crater_field.tri_height(z, sx, sy, pts[:, 0], pts[:, 1]) - ray).max()
            self.assertLess(err, 1e-4, f"{name}: 三角形の補間が MuJoCo と {err:.6f} m 違う")

    def test_crater_profile(self):
        """クレータの形：中心 −d、縁 +h で内外がつながり、半径の 3 倍で 0"""
        import crater_field

        p = crater_field.crater_profile(np.array([0.0, 1.0 - 1e-9, 1.0, 3.0 - 1e-9, 3.0, 5.0]), 1.0, 0.18)
        self.assertAlmostEqual(p[0], -1.0)
        self.assertAlmostEqual(p[1], 0.18, places=6)
        self.assertAlmostEqual(p[2], 0.18, places=6)
        self.assertAlmostEqual(p[3], 0.0, places=6)
        self.assertEqual(p[4], 0.0)
        self.assertEqual(p[5], 0.0)

    def test_craters_table(self):
        """クレータの配置表：数が平衡の期待値に合う・範囲と出発点の周り・形の比"""
        import crater_field

        for tile in TILES:
            name, grid, _ = load_tile(tile)
            path = tile / f"{name}.craters.json"
            if not path.exists():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            mo = doc["model"]
            c = np.array(doc["craters"], dtype=np.float64)
            self.assertEqual(len(c), doc["count"])
            half = grid["grid"]["half_extent_m"] - mo["margin_m"]
            self.assertTrue((np.abs(c[:, :2]) <= half + 1e-6).all())
            self.assertTrue((np.hypot(c[:, 0], c[:, 1]) >= mo["clear_radius_m"] + 3.0 * c[:, 2] / 2 - 1e-3).all(), "出発点の周りにクレータがある")
            self.assertTrue(((c[:, 2] >= mo["dmin_m"] - 1e-3) & (c[:, 2] <= mo["dmax_m"] + 1e-3)).all())
            ratio = c[:, 3] / c[:, 2]
            self.assertTrue(((ratio >= 0.03 - 1e-3) & (ratio <= 0.2 + 1e-3)).all(), "深さ / 直径が 0.03〜0.2 の外")
            self.assertTrue(np.allclose(c[:, 4], 0.18 * c[:, 3], atol=2e-4))
            # 出発点の周りを除く前の期待値に対し、ポアソンの揺らぎの 4 倍まで（除いた分は 1% 未満）
            expect = crater_field.equilibrium_count(mo["f_eq"], mo["dmin_m"], mo["dmax_m"], (2 * half) ** 2)
            self.assertLess(abs(doc["count"] - expect), 4.0 * math.sqrt(expect) + 0.01 * expect)
            # 大きさの分布：N(>D) ∝ D⁻² なので、D ≥ 2·dmin は全体のおよそ 1/4
            frac = float((c[:, 2] >= 2 * mo["dmin_m"]).mean())
            want = ((2 * mo["dmin_m"]) ** -2 - mo["dmax_m"] ** -2) / (mo["dmin_m"] ** -2 - mo["dmax_m"] ** -2)
            self.assertAlmostEqual(frac, want, delta=0.01)

    def test_fine_grid(self):
        """細かい格子：クレータから遠い点は 5 m の三角形の高さそのもの、クレータの中心は −d 前後へ下がる"""
        import crater_field

        for tile in TILES:
            name, grid, _ = load_tile(tile)
            if not (tile / f"{name}.craters.json").exists():
                continue
            meta, fine = crater_field.build_fine(tile)
            _, z = crater_field.load_coarse(tile)
            sx, sy = meta["half_extent_m"]
            res = meta["resolution_m"]
            self.assertEqual(fine.shape, (meta["nrow"], meta["ncol"]))
            # 原点は clear の中なので、原点の周りの点は 5 m の三角形と一致
            n0 = int(round(sx / res))
            ys, xs = np.mgrid[-10:11, -10:11]
            got = fine[n0 + ys, n0 + xs]
            want = crater_field.tri_height(z, sx, sy, xs * res, ys * res)
            self.assertLess(np.abs(got - want).max(), 1e-9)
            # 大きめのクレータの中心は、周りの 5 m の地面より d の 6 割以上下がる（重なりで多少ずれる）
            doc = json.loads((tile / f"{name}.craters.json").read_text(encoding="utf-8"))
            big = [q for q in doc["craters"] if q[2] >= 10.0][:50]
            for x, y, D, d, h in big:
                r, cidx = int(round((y + sy) / res)), int(round((x + sx) / res))
                drop = float(crater_field.tri_height(z, sx, sy, np.array(cidx * res - sx), np.array(r * res - sy))) - fine[r, cidx]
                self.assertGreater(drop, 0.6 * d, f"{name}: クレータ ({x:.1f}, {y:.1f}) の中心が下がっていない")

    def test_golombek_rapp(self):
        """k = 1% のとき、直径 0.2 m 以上の岩は約 0.007 個/m²（Golombek & Rapp 1997 の式を数値積分した値）"""
        import rock_field

        n, _, _ = rock_field.golombek_cumulative(0.01, np.array([0.2, 0.3, 0.5]))
        self.assertAlmostEqual(n[0], 0.00710, delta=0.0001)
        self.assertTrue(n[0] > n[1] > n[2] > 0)


if __name__ == "__main__":
    unittest.main()
