"""rock_field.py — 地形タイルに岩を撒き、物理と絵が共有する配置表を書き出す。

    python src/lunar_terrain/rock_field.py examples/datasets/lunar/lola_site01_t01 [--k 0.01] [--dmin 0.2] [--seed 25]

岩の大きさは Golombek–Rapp のモデル（岩が覆う面積の割合 F(>D) = k·exp(−q·D)、q = 1.79 + 0.152/k）から、
位置は一様（種を固定）。地面の高さと傾きは、タイルの HField（grid.json の mujoco）だけの MJCF を組み、
MuJoCo のレイで取る（その HField を地面にする物理と同じ高さ）。
岩の形は楕円体。物理は楕円体そのものを使い、絵は同じ楕円体の内側に凹凸を付けたメッシュで描く
（MuJoCo のメッシュは geom ごとに大きさを変えられないので、楕円体なら使う側が近くの岩だけ入れ替えて持てる）。

書き出し（タイルのフォルダ）:
  <tile>.rocks.json  配置表。rocks = [x, y, z, a, b, c, qw, qx, qy, qz, variant] の並び（grid.json の局所座標 m・z 上の右手系。
                     a, b, c は体の x, y, z の半径、四元数は体 → 世界、variant は絵のメッシュの番号）。
                     meshes = 絵のメッシュ（単位球の内側。半径 1 − bump..1。体の座標で楕円体の半径を掛ける）
物理の岩の持ち方（数・範囲・当たり方）は使う側が決める。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np


def golombek_cumulative(k: float, d: np.ndarray) -> np.ndarray:
    """直径 d 以上の岩の数 [個/m²]（F(>D) = k·exp(−qD) の面積を、1 個の面積 πD²/4 で割って積分）。"""
    q = 1.79 + 0.152 / k
    x = np.linspace(d.min(), 30.0, 400001)
    dens = k * q * np.exp(-q * x) / (math.pi * x * x / 4.0)
    cum = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) * 0.5 * np.diff(x))])
    total = cum[-1]
    return np.interp(d, x, total - cum), x, total - cum


def icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        mid, nf = {}, []

        def m(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                p = v[i] + v[j]
                v.append(p / np.linalg.norm(p))
                mid[key] = len(v) - 1
            return mid[key]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f, dtype=np.int32)


def rock_meshes(n: int, bump: float, rng: np.random.Generator) -> list[dict]:
    """凹凸のある単位球の内側のメッシュ n 種。半径は 1 − bump..1（物理の楕円体からはみ出さない）。"""
    v, f = icosphere(2)
    out = []
    for _ in range(n):
        # 低い周波数の凹凸：ランダムな向きの余弦の和を 0..1 に
        dirs = rng.normal(size=(6, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        s = sum(np.cos(rng.uniform(2, 5) * (v @ d) + rng.uniform(0, 2 * math.pi)) for d in dirs)
        s = (s - s.min()) / (s.max() - s.min())
        r = 1.0 - bump * s
        out.append({"vertices": [round(float(x), 5) for x in (v * r[:, None]).ravel()],
                    "triangles": [int(i) for i in f.ravel()]})
    return out


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, r.ravel())
    return q


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("tile", help="タイルのフォルダ（grid.json と hf.bin のある所）")
    ap.add_argument("--k", type=float, default=0.01, help="岩が覆う面積の割合（Golombek–Rapp の k。既定 1%%）")
    ap.add_argument("--dmin", type=float, default=0.2, help="撒く岩の最小の直径 [m]")
    ap.add_argument("--dmax", type=float, default=5.0, help="撒く岩の最大の直径 [m]")
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--clear", type=float, default=15.0, help="原点（ローバの出発点）の周りに岩を置かない半径 [m]")
    ap.add_argument("--margin", type=float, default=5.0, help="タイルの端から内側へ空ける幅 [m]")
    ap.add_argument("--variants", type=int, default=8, help="絵のメッシュの種類")
    ap.add_argument("--bump", type=float, default=0.05, help="絵の凹凸の深さ（楕円体の半径に対する割合）")
    a = ap.parse_args()
    tile = Path(a.tile).resolve()
    name = tile.name
    grid = json.loads((tile / f"{name}.grid.json").read_text(encoding="utf-8"))
    half = grid["grid"]["half_extent_m"] - a.margin
    rng = np.random.default_rng(a.seed)

    # 大きさ：累積の個数を逆に引いて直径を決める
    _, xs, cum = golombek_cumulative(a.k, np.array([a.dmin]))
    n_per_m2 = float(np.interp(a.dmin, xs, cum) - np.interp(a.dmax, xs, cum))
    area = (2 * half) ** 2 - math.pi * a.clear ** 2
    n = int(rng.poisson(n_per_m2 * area))
    u = rng.uniform(np.interp(a.dmax, xs, cum), np.interp(a.dmin, xs, cum), n)
    diam = np.interp(u, cum[::-1], xs[::-1])

    # 位置：一様。出発点の周りは空ける
    xy = rng.uniform(-half, half, (int(n * 1.2) + 10, 2))
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) > a.clear][:n]

    # 地面の高さと法線：タイルの HField だけの MJCF へ鉛直にレイ（中心と ±0.5 m の 3 点で面を取る）
    mj = grid["mujoco"]
    sx, sy, sz, base = mj["size"]
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><hfield name="dem" file="{(tile / mj["file"]).as_posix()}"'
        f' size="{sx:.3f} {sy:.3f} {sz:.4f} {base:.3f}"/></asset>'
        f'<worldbody><geom type="hfield" hfield="dem" pos="0 0 {mj["geom_pos_z"]:.4f}" group="1"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    groups[1] = 1
    gid = np.zeros(1, dtype=np.int32)
    top = grid["grid"]["z_local_max_m"] + 50.0

    def ground(x, y):
        dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0, 0, -1.0]), groups, 1, -1, gid)
        return top - dist

    rocks = []
    for i in range(n):
        x, y = xy[i]
        z = ground(x, y)
        h = 0.5
        nrm = np.cross([2 * h, 0, ground(x + h, y) - ground(x - h, y)], [0, 2 * h, ground(x, y + h) - ground(x, y - h)])
        nrm /= np.linalg.norm(nrm)
        # 平面の大きさ：直径 D は水平の 2 軸の平均。縦横比 1..1.4、高さ（地面から出ている分）は 0.5·D ±20%
        s = rng.uniform(1.0, 1.4) ** 0.5
        ra, rb = diam[i] / 2 * s, diam[i] / 2 / s
        exposed = 0.5 * diam[i] * rng.uniform(0.8, 1.2)
        sink = rng.uniform(-0.3, 0.1)                 # 中心の高さ / c（負なら中心は地面の下）
        rc = exposed / (1.0 + sink)
        # 向き：z を地面の法線に、ヨーは一様、さらに 10° までの傾き
        yaw = rng.uniform(0, 2 * math.pi)
        tilt_ax = rng.normal(size=3)
        tilt_ax -= tilt_ax @ nrm * nrm
        tilt_ax /= np.linalg.norm(tilt_ax)
        zax = nrm * math.cos(math.radians(rng.uniform(0, 10))) + tilt_ax * math.sin(math.radians(rng.uniform(0, 10)))
        zax /= np.linalg.norm(zax)
        ref = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        xax = ref - ref @ zax * zax
        xax /= np.linalg.norm(xax)
        yax = np.cross(zax, xax)
        rot = np.column_stack([xax, yax, zax])
        q = quat_from_matrix(rot)
        c = np.array([x, y, z]) + nrm * (sink * rc)
        rocks.append([round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4),
                      round(float(ra), 4), round(float(rb), 4), round(float(rc), 4),
                      round(float(q[0]), 6), round(float(q[1]), 6), round(float(q[2]), 6), round(float(q[3]), 6),
                      int(rng.integers(a.variants))])

    doc = {
        "name": name,
        "generator": "hakoniwa-simenv-data src/lunar_terrain/rock_field.py",
        "frame": "grid.json の局所座標（右手系 m・z 上）",
        "model": {"size_frequency": "Golombek & Rapp (1997), J. Geophys. Res. 102(E2), 4117-4129: F(>D) = k·exp(-qD), q = 1.79 + 0.152/k",
                  "k": a.k, "dmin_m": a.dmin, "dmax_m": a.dmax, "seed": a.seed, "clear_radius_m": a.clear,
                  "margin_m": a.margin, "shape": "楕円体。D = 水平 2 軸の平均の直径、地面から出ている高さ 0.5·D ±20%"},
        "fields": ["x", "y", "z", "a", "b", "c", "qw", "qx", "qy", "qz", "variant"],
        "count": len(rocks),
        "rocks": rocks,
        "meshes": rock_meshes(a.variants, a.bump, rng),
        "mesh_note": f"単位球の内側（半径 {1 - a.bump:.2f}..1）。頂点 (x, y, z)·(a, b, c) を岩の体の座標とする",
    }
    (tile / f"{name}.rocks.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    # 枠は 1 つずつ mocap の body に入れる（実行中に位置・向きを変えてよい作り。世界の body の geom を動かすと、
    # コンパイル時に作られる body の当たり判定の木と食い違う）
    dd = np.array([r[3] + r[4] for r in rocks])
    hh = np.array([r[5] for r in rocks])
    print(f"[rock_field] {name}: k={a.k} D={a.dmin}..{a.dmax} m  岩 {len(rocks)} 個（期待値 {n_per_m2 * area:.0f}）"
          f"  D≥0.3 m {int((dd >= 0.3).sum())}・D≥0.5 m {int((dd >= 0.5).sum())}・最大 D {dd.max():.2f} m・最大の c {hh.max():.2f} m")
    print(f"[rock_field] → {tile / (name + '.rocks.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
