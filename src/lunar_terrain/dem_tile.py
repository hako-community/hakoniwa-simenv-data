"""dem_tile.py — 月の極域 DEM（GeoTIFF・極投影）から 1 タイルを切り出し、物理と絵の地形を作る。

    python src/lunar_terrain/dem_tile.py --name lola_site01_t01 --center -16250 -11750 --size 2000
      （入力は dem.lock.json の 1 件目を dem_fetch.py で取ったもの。--src で別の GeoTIFF も可）

作るもの（examples/datasets/lunar/<name>/。--out-dir で変えられる）:
  <name>.grid.json   共通標高格子の契約（投影・原点・解像度・高さのずらし・北の向き・出典）
  <name>.hf.bin      MuJoCo の HField（int32 nrow, int32 ncol, float32 高さ[m] 行優先。行 0 = 南端 −y）
  <name>.height16.png  Unreal Landscape 用の 16bit 高さ画像（上の行 = +y）。倍率は grid.json の unreal
世界の MJCF（車両を include し、HField を地面にするもの）は使う側が grid.json の mujoco から作る。

座標の約束:
  局所座標（FLU と同じ向きの世界座標）= 極投影の (X − X0, Y − Y0)、高さ = 標高 − Z0（原点の標高を 0 にする）。
  ★ 極投影の +Y は「北」ではない。原点での北（緯度が増える向き。南極では極から遠ざかる）は grid.json の north_dir_local。
  ★ 極投影の縮尺のずれ（南緯 89° で 1.0001 程度）は 2 km で 0.2 m 未満なので無視する（grid.json に記録）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--src", help="GeoTIFF（既定：dem.lock.json の 1 件目）")
    ap.add_argument("--out-dir", default=str(ROOT / "examples" / "datasets" / "lunar"), help="タイルのフォルダを作る場所")
    ap.add_argument("--center", type=float, nargs=2, required=True, metavar=("X", "Y"), help="極投影の中心 [m]")
    ap.add_argument("--size", type=float, default=2000.0, help="一辺 [m]")
    ap.add_argument("--base", type=float, default=10.0, help="HField の底の厚み [m]")
    a = ap.parse_args()

    lock = json.loads((HERE / "dem.lock.json").read_text(encoding="utf-8"))
    entry = lock["files"][0] if not a.src else None
    src = Path(a.src) if a.src else ROOT / "deps" / "dem" / entry["path"]
    if entry and sha256(src) != entry["sha256"]:
        raise SystemExit(f"SHA256 が dem.lock と違います: {src}（dem_fetch.py で取り直す）")

    with rasterio.open(src) as r:
        res = r.transform.a
        assert abs(r.transform.e + res) < 1e-9 and r.transform.b == 0 and r.transform.d == 0, "北上の正方格子だけ扱う"
        n = int(round(a.size / res)) + 1                              # 点の数（端を含む）
        col_c, row_c = ~r.transform * (a.center[0], a.center[1])     # 画素の角基準の座標
        col0 = int(round(col_c - 0.5 - (n - 1) / 2))
        row0 = int(round(row_c - 0.5 - (n - 1) / 2))
        if col0 < 0 or row0 < 0 or col0 + n > r.width or row0 + n > r.height:
            raise SystemExit("切り出しが DEM の外に出ます")
        z = r.read(1, window=Window(col0, row0, n, n)).astype(np.float64)   # 行 0 = 北（+Y）
        if not np.isfinite(z).all():
            raise SystemExit("欠損があります")
        # 画素中心の極投影座標（README：pixel-registered = 値は画素の中心）
        x0, y0 = r.transform * (col0 + 0.5 + (n - 1) / 2, row0 + 0.5 + (n - 1) / 2)
        crs = r.crs
        crs_text = crs.to_proj4()

    zc = float(z[(n - 1) // 2, (n - 1) // 2])
    zl = z - zc                                                      # 原点の標高を 0 に
    zmin, zmax = float(zl.min()), float(zl.max())
    half = (n - 1) * res / 2.0
    lon, lat = warp_transform(crs, "+proj=longlat +R=1737400 +no_defs", [x0], [y0])
    # 北 = 緯度が増える向き（南極の投影では極から遠ざかる向き）。投影の式に頼らず、少し北の点への向きで出す
    xn, yn = warp_transform("+proj=longlat +R=1737400 +no_defs", crs, [lon[0]], [lat[0] + 0.01])
    dn = math.hypot(xn[0] - x0, yn[0] - y0)
    north = [(xn[0] - x0) / dn, (yn[0] - y0) / dn]
    k = 2.0 / (1.0 + math.sin(math.radians(abs(lat[0]))))            # 球の極投影の縮尺（極で 1）
    gy, gx = np.gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    # MuJoCo：行 0 = −y（南端）なので上下を反転して書く
    hf = np.flipud(zl).astype(np.float32)
    with (out / f"{a.name}.hf.bin").open("wb") as f:
        np.array([n, n], dtype=np.int32).tofile(f)
        hf.tofile(f)
    # Unreal：16bit・上の行 = +y。0..65535 を zmin..zmax に割り当てる
    from PIL import Image
    q = np.round((zl - zmin) / (zmax - zmin) * 65535.0).astype(np.uint16)
    Image.fromarray(q).save(out / f"{a.name}.height16.png")

    grid = {
        "name": a.name,
        "generator": "hakoniwa-simenv-data src/lunar_terrain/dem_tile.py",
        "source": {"file": src.name, "sha256": sha256(src), "url": entry["url"] if entry else None,
                   "citation": entry.get("citation") if entry else None,
                   "crs": crs_text, "window_rowcol": [row0, col0], "pixel_registration": "center"},
        "origin": {"stereo_xy_m": [x0, y0], "lat_deg": lat[0], "lon_deg": lon[0], "elevation_m": zc},
        "grid": {"n": n, "resolution_m": res, "half_extent_m": half, "z_local_min_m": zmin, "z_local_max_m": zmax},
        "frame": "local = (X - X0, Y - Y0, Z - Z0)。x, y は極投影の向き（FLU の世界座標として使う）",
        "north_dir_local": north,
        "stereo_scale_factor": k,
        "slope_deg": {"mean": float(slope.mean()), "p95": float(np.percentile(slope, 95)), "max": float(slope.max()),
                      "frac_10_20": float(((slope >= 10) & (slope < 20)).mean()), "frac_ge_20": float((slope >= 20).mean())},
        "mujoco": {"file": f"{a.name}.hf.bin", "size": [half, half, zmax - zmin, a.base], "geom_pos_z": zmin,
                   "row0": "south (-y)"},
        "unreal": {"file": f"{a.name}.height16.png", "row0": "north (+y)", "points": n,
                   "xy_scale_cm": res * 100.0, "z_scale_percent": (zmax - zmin) * 100.0 * 128.0 / 65535.0,
                   "location_z_cm": (zmin + (zmax - zmin) * 32768.0 / 65535.0) * 100.0,
                   "note": "高さ[cm] = location_z + (値 − 32768)·z_scale/128。値 0 → z_local_min、65535 → z_local_max"},
    }
    (out / f"{a.name}.grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"[dem_tile] {out}  n={n} res={res} m  z={zmin:.1f}..{zmax:.1f} m  lat={lat[0]:.3f} lon={lon[0]:.2f}")
    print(f"[dem_tile] slope mean={grid['slope_deg']['mean']:.1f} p95={grid['slope_deg']['p95']:.1f} 10-20deg={grid['slope_deg']['frac_10_20']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
