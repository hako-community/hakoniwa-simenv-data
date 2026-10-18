"""sun_ephem.py — 地形タイルの原点から見た太陽・地球の方位と高度（位置暦）を出す。

    python src/lunar_terrain/sun_ephem.py --fetch                       # カーネルを deps/ephem/ に取る（SHA256 照合）
    python src/lunar_terrain/sun_ephem.py examples/datasets/lunar/lola_site01_t01 --scan 2026-10-01 30   # 30 日を 6 時間ごとに表で見る
    python src/lunar_terrain/sun_ephem.py examples/datasets/lunar/lola_site01_t01 --utc 2026-10-24T12:00:00Z   # <name>.sun.json を書く

計算：Skyfield ＋ NAIF のカーネル（ephem.lock.json。DE440・月の ME 座標 MOON_ME_DE440_ME421）。
  原点の緯度・経度・標高（grid.json の origin）に立ち、太陽・地球を光行差・光の時間つきで見た高度・方位を出す。
  方位は北から東回り。局所座標（grid.json の frame）の向きへは north_dir_local で回す：
    東 = 北を上から見て時計回りに 90°（局所座標は右手系・z 上）
    向き = cos(高度)·(cos(方位)·北 + sin(方位)·東) + sin(高度)·z     ← 原点から天体へ向かう単位ベクトル
  高度は月の球（R = 1737.4 km）の接平面から。地形の起伏（タイルの外の山を含む）による日陰は描画側で決まる。

書き出し（<タイルのフォルダ>/<name>.sun.json）：時刻・太陽と地球の高度・方位・局所座標の向き・距離・視直径。
  ビューア（Unreal・Godot）はこの向きで平行光源と地球を置く。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = HERE / "ephem.lock.json"
DEPS = ROOT / "deps" / "ephem"
ME_FRAME = "MOON_ME_DE440_ME421"
SUN_RADIUS_KM = 695700.0
EARTH_RADIUS_KM = 6371.0
AU_KM = 149597870.7
SOLAR_LUX_1AU = 128000.0      # 大気の外の太陽の照度（1361 W/m² × 約 94 lm/W）
EARTH_BOND_ALBEDO = 0.30


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for e in lock["files"]:
        dst = DEPS / e["path"]
        if dst.exists() and sha256(dst) == e["sha256"]:
            print(f"[sun_ephem] ok   {dst.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[sun_ephem] get  {e['url']}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(e["url"], tmp)
        if sha256(tmp) != e["sha256"]:
            tmp.unlink()
            raise SystemExit(f"SHA256 が違います: {e['url']}")
        tmp.replace(dst)
        print(f"[sun_ephem] done {dst.name}")


class Site:
    """タイルの原点に立つ観測者。"""

    def __init__(self, grid: dict):
        from skyfield.api import Loader
        from skyfield.planetarylib import PlanetaryConstants

        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        for e in lock["files"]:
            p = DEPS / e["path"]
            if not p.exists() or sha256(p) != e["sha256"]:
                raise SystemExit(f"カーネルが無いか版が違います: {p}（--fetch で取り直す）")
        load = Loader(str(DEPS))
        self.ts = load.timescale(builtin=True)
        self.eph = load("de440s.bsp")
        pc = PlanetaryConstants()
        pc.read_text(load("moon_de440_250416.tf"))
        pc.read_text(load("pck00011.tpc"))
        pc.read_binary(load("moon_pa_de440_200625.bpc"))
        # ★ Skyfield は .bpc の区間を体ごとに最後の 1 つしか持たない（DE440 の月の向きは約 225 年ごとの区間）。
        #   時刻を含む区間を選んで、区間ごとに観測者を組み立てる
        from jplephem.pck import PCK
        self.segments = list(PCK.open(str(DEPS / "moon_pa_de440_200625.bpc")).segments)
        self.pc = pc
        self.observers = {}
        o = grid["origin"]
        self.lat, self.lon, self.h = o["lat_deg"], o["lon_deg"], o["elevation_m"]
        self.north = check_north(grid)
        self.east = [self.north[1], -self.north[0]]

    def observer(self, t):
        jd = t.tdb
        seg = next((s for s in self.segments if s.initial_jd <= jd <= s.final_jd), None)
        if seg is None:
            raise SystemExit(f"月の向きのカーネルの範囲外の時刻です: {t.utc_iso()}")
        key = (seg.body, seg.initial_jd)
        if key not in self.observers:
            self.pc._segment_map[seg.body] = seg
            frame = self.pc.build_frame_named(ME_FRAME)
            self.observers[key] = self.eph["moon"] + self.pc.build_latlon_degrees(frame, self.lat, self.lon, self.h)
        return self.observers[key]

    def look(self, when: datetime, body: str) -> dict:
        t = self.ts.from_datetime(when)
        alt, az, dist = self.observer(t).at(t).observe(self.eph[body]).apparent().altaz()
        el, a = alt.radians, az.radians
        h = math.cos(el)
        d = [h * (math.cos(a) * self.north[i] + math.sin(a) * self.east[i]) for i in range(2)] + [math.sin(el)]
        r = SUN_RADIUS_KM if body == "sun" else EARTH_RADIUS_KM
        return {"elevation_deg": alt.degrees, "azimuth_deg": az.degrees, "dir_local": d,
                "distance_km": dist.km, "angular_diameter_deg": math.degrees(2.0 * math.asin(r / dist.km))}


def check_north(grid: dict) -> list[float]:
    """grid.json の north_dir_local を、緯度を少し上げた点への向き（極投影で独立に計算）と照らす。"""
    from rasterio.warp import transform as warp_transform

    o = grid["origin"]
    crs = grid["source"]["crs"]
    x0, y0 = o["stereo_xy_m"]
    xs, ys = warp_transform("+proj=longlat +R=1737400 +no_defs", crs, [o["lon_deg"]], [o["lat_deg"] + 0.01])
    dx, dy = xs[0] - x0, ys[0] - y0
    n = math.hypot(dx, dy)
    calc = [dx / n, dy / n]
    north = grid["north_dir_local"]
    if calc[0] * north[0] + calc[1] * north[1] < 0.9999:
        raise SystemExit(f"grid.json の north_dir_local {north} が緯度の増える向き {calc} と合いません（dem_tile.py で作り直す）")
    return north


def add_light(s: dict, e: dict) -> None:
    """照度を足す。太陽：大気の外の 1 AU で SOLAR_LUX_1AU を距離の 2 乗で。
    地球照：地球をランバート球（球面アルベド EARTH_BOND_ALBEDO）とし、位相角 α の明るさ Φ(α) = (sin α + (π − α) cos α) / π。
      E = (2/3)·A·E_sun·(R/d)²·Φ(α)。満地球で約 7 lux（太陽の 2 万分の 1 弱。影の中だけで効く）"""
    s["illuminance_lux"] = SOLAR_LUX_1AU * (AU_KM / s["distance_km"]) ** 2
    cos_el = sum(p * q for p, q in zip(s["dir_local"], e["dir_local"]))   # 月から見た太陽と地球の離角
    alpha = math.pi - math.acos(max(-1.0, min(1.0, cos_el)))              # 地球での位相角（太陽は十分遠い）
    phi = (math.sin(alpha) + (math.pi - alpha) * math.cos(alpha)) / math.pi
    e["phase_angle_deg"] = math.degrees(alpha)
    e["illuminated_fraction"] = (1.0 + math.cos(alpha)) / 2.0
    e["earthshine_lux"] = (2.0 / 3.0) * EARTH_BOND_ALBEDO * s["illuminance_lux"] * (EARTH_RADIUS_KM / e["distance_km"]) ** 2 * phi
    e["earthshine_note"] = f"ランバート球・アルベド {EARTH_BOND_ALBEDO}・太陽 {SOLAR_LUX_1AU:.0f} lux at 1 AU の近似"


def parse_utc(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)   # 時差なしは UTC


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("tile", nargs="?", help="タイルのフォルダ（grid.json のあるフォルダ。例 examples/datasets/lunar/<name>）")
    ap.add_argument("--fetch", action="store_true", help="カーネルを取る")
    ap.add_argument("--utc", help="時刻（ISO 8601・UTC）。<name>.sun.json を書く")
    ap.add_argument("--scan", nargs=2, metavar=("START", "DAYS"), help="START から DAYS 日の表")
    ap.add_argument("--step-h", type=float, default=6.0, help="--scan の刻み [h]")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        if not a.tile:
            return 0
    if not a.tile:
        ap.error("tile が要ります")
    tile = Path(a.tile)
    grid = json.loads((tile / f"{tile.name}.grid.json").read_text(encoding="utf-8"))
    site = Site(grid)
    print(f"[sun_ephem] {tile.name}  南緯 {abs(site.lat):.3f}°  経度 {site.lon:.2f}°  北（局所）= ({site.north[0]:+.3f}, {site.north[1]:+.3f})")

    if a.scan:
        t0 = parse_utc(a.scan[0])
        n = int(float(a.scan[1]) * 24.0 / a.step_h) + 1
        print("  UTC                 太陽 高度   方位   地球 高度   方位")
        for i in range(n):
            t = t0 + timedelta(hours=a.step_h * i)
            s, e = site.look(t, "sun"), site.look(t, "earth")
            print(f"  {t:%Y-%m-%dT%H:%MZ}   {s['elevation_deg']:+6.2f}° {s['azimuth_deg']:6.1f}°    {e['elevation_deg']:+6.2f}° {e['azimuth_deg']:6.1f}°")

    if a.utc:
        t = parse_utc(a.utc)
        s, e = site.look(t, "sun"), site.look(t, "earth")
        add_light(s, e)
        out = {
            "name": grid["name"],
            "generator": "hakoniwa-simenv-data src/lunar_terrain/sun_ephem.py",
            "utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observer": {"lat_deg": site.lat, "lon_deg": site.lon, "elevation_m": site.h, "frame": ME_FRAME},
            "kernels": [e_["path"] for e_ in json.loads(LOCK.read_text(encoding="utf-8"))["files"]],
            "north_dir_local": site.north,
            "frame": "dir_local は grid.json の局所座標（右手系・z 上）で、原点から天体へ向かう単位ベクトル。方位は北から東回り",
            "sun": s,
            "earth": e,
        }
        dst = tile / f"{tile.name}.sun.json"
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[sun_ephem] {out['utc']}  太陽 高度 {s['elevation_deg']:+.2f}° 方位 {s['azimuth_deg']:.1f}°"
              f"  地球 高度 {e['elevation_deg']:+.2f}° 方位 {e['azimuth_deg']:.1f}°")
        print(f"[sun_ephem] 太陽の向き（局所）= ({s['dir_local'][0]:+.4f}, {s['dir_local'][1]:+.4f}, {s['dir_local'][2]:+.4f})"
              f"  照度 {s['illuminance_lux']:.0f} lux  地球の位相角 {e['phase_angle_deg']:.0f}°（輝面 {e['illuminated_fraction']:.0%}）"
              f"・地球照 {e['earthshine_lux']:.2f} lux  → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
