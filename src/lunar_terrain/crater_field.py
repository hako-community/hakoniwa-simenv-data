"""crater_field.py — 地形タイルに直径数 m〜20 m の小クレータを撒き、物理と絵が共有する配置表と細かい格子を作る。

    python src/lunar_terrain/crater_field.py examples/datasets/lunar/lola_site01_t01 [--f-eq 0.03] [--dmin 2] [--dmax 20] [--seed 28]
    python src/lunar_terrain/crater_field.py examples/datasets/lunar/lola_site01_t01 --fine     # 細かい格子を作る（git 管理外）

5 m の DEM（L1）では表せない小クレータ（L2）を、サイズ頻度分布から合成する（実測の細かい地形 LROC NAC DTM は
このタイルの場所に無い）。
  数：小さいクレータは「平衡」（新しくできる数と埋もれて消える数がつり合う）にあるとし、累積の数を
      N(>D) = f_eq · 1.54 · D⁻² [個/m²]（D [m]、1.54 D⁻² は幾何学的な飽和、f_eq はその割合）とする。
      Gault (1970) は f_eq を 1〜10%、Xiao & Werner (2015) は 0.7〜3.9% と見積もった。既定は 3%。
  形：深さ d / 直径 D は 0.03〜0.2 の対数一様（新しいクレータは Pike (1977) の 0.2、古いものほど浅い）。
      縁の高さ h = 0.18·d（新しいクレータの縁 0.036 D と深さ 0.2 D の比）。半径 R、中心からの距離を r·R として
        r < 1   ：−d + (d + h)·r²（お椀）
        1 ≤ r < 3：h·(r⁻³ − 1/27)/(1 − 1/27)（縁の外へ減る。r = 3 で 0）
      重なったクレータは足し合わせる。
高さの約束：細かい格子（既定 0.5 m）の各点 = 5 m の HField を MuJoCo と同じ三角形で補間した高さ ＋ クレータの足し合わせ。
物理・絵はこの格子を同じ向きの三角形（各マスを (0,0)–(1,1) の対角で切る。MuJoCo の HField と同じ）で使う。

書き出し（タイルのフォルダ）:
  <tile>.craters.json   配置表（git 管理）。craters = [x, y, D, d, h] の並び（grid.json の局所座標 m）
  --fine のとき（git 管理外。配置表と hf.bin から決まるので、使う側が作り直す）:
  <tile>.fine.hf.bin    細かい格子（MuJoCo の HField のバイナリ：int32 行数・列数 ＋ float32 の局所の高さ。行 0 = 南 −y）
  <tile>.fine.json      格子の大きさ・高さの範囲・材料の SHA256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SAT_COEF = 1.54            # 幾何学的な飽和の累積の数 N(>D) = 1.54 D⁻²
PROFILE_OUTER = 3.0        # 縁の外の裾は半径の 3 倍まで


def load_coarse(tile: Path) -> tuple[dict, np.ndarray]:
    """grid.json と 5 m の HField（局所の高さ。行 0 = 南）を読む。"""
    name = tile.name
    grid = json.loads((tile / f"{name}.grid.json").read_text(encoding="utf-8"))
    path = tile / grid["mujoco"]["file"]
    nrow, ncol = (int(v) for v in np.fromfile(path, dtype=np.int32, count=2))
    z = np.fromfile(path, dtype=np.float32, offset=8).reshape(nrow, ncol).astype(np.float64)
    return grid, z


def tri_height(z: np.ndarray, half_x: float, half_y: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MuJoCo の HField と同じ三角形の補間（列 = x、行 = y、各マスを (0,0)–(1,1) の対角で切る）。"""
    nrow, ncol = z.shape
    u = (np.asarray(x) + half_x) / (2.0 * half_x) * (ncol - 1)
    v = (np.asarray(y) + half_y) / (2.0 * half_y) * (nrow - 1)
    c = np.clip(np.floor(u).astype(np.int64), 0, ncol - 2)
    r = np.clip(np.floor(v).astype(np.int64), 0, nrow - 2)
    fu, fv = u - c, v - r
    z00, z10, z01, z11 = z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]
    za = z00 + fu * (z10 - z00) + fv * (z11 - z10)
    zb = z00 + fv * (z01 - z00) + fu * (z11 - z01)
    return np.where(fu >= fv, za, zb)


def crater_profile(rr: np.ndarray, d: float, h: float) -> np.ndarray:
    """半径で割った距離 rr でのクレータの高さ（周りの地面からの差）。"""
    inner = -d + (d + h) * rr * rr
    k = PROFILE_OUTER ** -3
    outer = h * (np.power(np.maximum(rr, 1.0), -3.0) - k) / (1.0 - k)
    return np.where(rr < 1.0, inner, np.where(rr < PROFILE_OUTER, outer, 0.0))


def equilibrium_count(f_eq: float, dmin: float, dmax: float, area_m2: float) -> float:
    """dmin..dmax のクレータの数の期待値。"""
    a = f_eq * SAT_COEF
    return a * (dmin ** -2 - dmax ** -2) * area_m2


def generate(grid: dict, f_eq: float, dmin: float, dmax: float, seed: int, clear: float, margin: float) -> list[list[float]]:
    half = grid["grid"]["half_extent_m"] - margin
    rng = np.random.default_rng(seed)
    area = (2 * half) ** 2
    n = int(rng.poisson(equilibrium_count(f_eq, dmin, dmax, area)))
    # 直径：N(>D) ∝ D⁻² を dmin..dmax で切った分布を逆関数で引く
    u = rng.uniform(0.0, 1.0, n)
    diam = (dmin ** -2 - u * (dmin ** -2 - dmax ** -2)) ** -0.5
    xy = rng.uniform(-half, half, (n, 2))
    dd = np.exp(rng.uniform(math.log(0.03), math.log(0.2), n))
    out = []
    for i in range(n):
        R = diam[i] / 2.0
        # 出発点（原点）の周りの clear [m] には裾もかからないようにする
        if math.hypot(xy[i, 0], xy[i, 1]) < clear + PROFILE_OUTER * R:
            continue
        d = dd[i] * diam[i]
        out.append([round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3), round(float(diam[i]), 3),
                    round(float(d), 4), round(float(0.18 * d), 4)])
    return out


def build_fine(tile: Path, res: float | None = None) -> tuple[dict, np.ndarray]:
    """細かい格子（局所の高さ。行 0 = 南）を作る。配置表が無ければ 5 m の HField を細かく補間しただけ。"""
    grid, z = load_coarse(tile)
    name = tile.name
    craters_path = tile / f"{name}.craters.json"
    doc = json.loads(craters_path.read_text(encoding="utf-8")) if craters_path.exists() else {"craters": [], "fine": {"resolution_m": 0.5}}
    res = float(res or doc["fine"]["resolution_m"])
    sx, sy = grid["mujoco"]["size"][0], grid["mujoco"]["size"][1]
    nx, ny = int(round(2 * sx / res)) + 1, int(round(2 * sy / res)) + 1
    xs = np.linspace(-sx, sx, nx)
    ys = np.linspace(-sy, sy, ny)
    fine = np.empty((ny, nx), dtype=np.float64)
    for r0 in range(0, ny, 256):
        yy, xx = np.meshgrid(ys[r0:r0 + 256], xs, indexing="ij")
        fine[r0:r0 + 256] = tri_height(z, sx, sy, xx, yy)
    for x, y, D, d, h in doc["craters"]:
        R = D / 2.0
        reach = PROFILE_OUTER * R
        c0, c1 = max(0, int(math.floor((x - reach + sx) / res))), min(nx, int(math.ceil((x + reach + sx) / res)) + 1)
        r0, r1 = max(0, int(math.floor((y - reach + sy) / res))), min(ny, int(math.ceil((y + reach + sy) / res)) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        yy, xx = np.meshgrid(ys[r0:r1], xs[c0:c1], indexing="ij")
        fine[r0:r1, c0:c1] += crater_profile(np.hypot(xx - x, yy - y) / R, d, h)
    meta = {"resolution_m": res, "nrow": ny, "ncol": nx, "half_extent_m": [sx, sy],
            "z_local_min_m": float(fine.min()), "z_local_max_m": float(fine.max()), "row0": "south (-y)",
            "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (tile / grid["mujoco"]["file"], craters_path) if p.exists()}}
    return meta, fine


def write_fine(tile: Path, meta: dict, fine: np.ndarray) -> Path:
    name = tile.name
    path = tile / f"{name}.fine.hf.bin"
    with open(path, "wb") as f:
        np.array([meta["nrow"], meta["ncol"]], dtype=np.int32).tofile(f)
        fine.astype(np.float32).tofile(f)
    meta = dict(meta, file=path.name)
    (tile / f"{name}.fine.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("tile", help="タイルのフォルダ（grid.json と hf.bin のある所）")
    ap.add_argument("--f-eq", type=float, default=0.03, help="平衡の数の、幾何学的な飽和に対する割合（既定 3%%）")
    ap.add_argument("--dmin", type=float, default=2.0, help="撒くクレータの最小の直径 [m]（格子の 4 点分）")
    ap.add_argument("--dmax", type=float, default=20.0, help="撒くクレータの最大の直径 [m]（それより大きいものは DEM にある）")
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--clear", type=float, default=15.0, help="原点（出発点）の周りにクレータを置かない半径 [m]")
    ap.add_argument("--margin", type=float, default=5.0, help="タイルの端から内側へ空ける幅 [m]")
    ap.add_argument("--res", type=float, default=0.5, help="細かい格子の間隔 [m]")
    ap.add_argument("--fine", action="store_true", help="配置表から細かい格子を作る（配置表は作り直さない）")
    a = ap.parse_args()
    tile = Path(a.tile).resolve()
    name = tile.name
    if a.fine:
        meta, fine = build_fine(tile, a.res)
        path = write_fine(tile, meta, fine)
        print(f"[crater_field] fine {meta['ncol']}×{meta['nrow']} @ {meta['resolution_m']} m  z {meta['z_local_min_m']:.2f}..{meta['z_local_max_m']:.2f}  → {path}")
        return 0
    grid, _ = load_coarse(tile)
    craters = generate(grid, a.f_eq, a.dmin, a.dmax, a.seed, a.clear, a.margin)
    half = grid["grid"]["half_extent_m"] - a.margin
    doc = {
        "name": name,
        "generator": "hakoniwa-simenv-data src/lunar_terrain/crater_field.py",
        "frame": "grid.json の局所座標（右手系 m・z 上）",
        "model": {
            "size_frequency": "equilibrium N(>D) = f_eq * 1.54 * D^-2 per m^2 (D in m); Gault (1970) f_eq 1-10%, Xiao & Werner (2015) 0.7-3.9%",
            "f_eq": a.f_eq, "dmin_m": a.dmin, "dmax_m": a.dmax, "seed": a.seed, "clear_radius_m": a.clear, "margin_m": a.margin,
            "shape": "d/D log-uniform 0.03-0.2 (fresh 0.2: Pike 1977), rim h = 0.18 d; bowl -d + (d+h) r^2 (r<1), rim h (r^-3 - 1/27)/(1 - 1/27) (1<=r<3)",
            "expected_count": equilibrium_count(a.f_eq, a.dmin, a.dmax, (2 * half) ** 2),
        },
        "fields": ["x", "y", "D", "d", "h"],
        "count": len(craters),
        "craters": craters,
        "fine": {"resolution_m": a.res, "triangulation": "MuJoCo HField: each cell split along (0,0)-(1,1)",
                 "height": "tri-interpolated 5 m HField + sum of crater profiles"},
    }
    (tile / f"{name}.craters.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    dd = np.array([c[2] for c in craters])
    print(f"[crater_field] {name}: f_eq={a.f_eq} D={a.dmin}..{a.dmax} m  クレータ {len(craters)} 個（期待値 {doc['model']['expected_count']:.0f}）"
          f"  D≥5 m {int((dd >= 5).sum())}・D≥10 m {int((dd >= 10).sum())}・最大 D {dd.max():.1f} m")
    print(f"[crater_field] → {tile / (name + '.craters.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
