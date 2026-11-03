#!/usr/bin/env python3
"""海域プリセット（scenarios/sea_areas/*.json）の幾何（SHP-28、ADR D36）。

- 航路の区域：法令の定義（基準点からの真方位・距離）→ 局所 ENU の多角形。座標の換算は物理の include/ship/geo/local_enu.hpp と同じ
  正距円筒近似（原点の緯度、球 R = 6378137 m）。数 km〜十数 km の海域で、法令の測地線の距離との差は 0.1 % 未満
- 航路の中心線・通航帯（右側通航：正の向きに進む船は中心線の右）
- 潮流の場（航路の軸に沿う往復流、航路の外は距離で弱まる）：物理（C++）にも同じ式を置く。ここは試験・図・生成器用

  python scenarios/sea_area.py scenarios/sea_areas/uraga_suido.json     # 多角形・長さ・幅を表示
"""
import json
import math
import sys

EARTH_R = 6378137.0
KT = 0.514444


class GeoOrigin:
    def __init__(self, lat_deg, lon_deg):
        self.lat = lat_deg
        self.lon = lon_deg

    def to_enu(self, lat, lon):
        lat0 = math.radians(self.lat)
        return (math.radians(lon - self.lon) * EARTH_R * math.cos(lat0), math.radians(lat - self.lat) * EARTH_R)

    def to_geo(self, east, north):
        lat0 = math.radians(self.lat)
        return (self.lat + math.degrees(north / EARTH_R), self.lon + math.degrees(east / (EARTH_R * math.cos(lat0))))


def offset(p, bearing_deg, dist_m):
    b = math.radians(bearing_deg)
    return (p[0] + dist_m * math.sin(b), p[1] + dist_m * math.cos(b))


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def dist_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def dist_to_polygon(p, poly):
    if point_in_polygon(p, poly):
        return 0.0
    return min(dist_to_segment(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


class Route:
    """航路：多角形と、両端の辺（start・end）から作る中心線"""

    def __init__(self, z, refs_enu):
        self.name = z["name"]
        self.keep_right = z.get("keep_right", True)
        self.speed_limit_kt = z.get("speed_limit_kt")
        self.poly = [offset(refs_enu[p["ref"]], p["bearing_deg"], p["dist_m"]) for p in z["points"]]
        s, e = z["ends"]["start"], z["ends"]["end"]
        self.start_edge = (self.poly[s[0]], self.poly[s[1]])
        self.end_edge = (self.poly[e[0]], self.poly[e[1]])
        # 両側の境界：start の辺から end の辺へ、多角形を順にたどる 2 本
        n = len(self.poly)
        def walk(a, b, step):
            out, i = [a], a
            while i != b:
                i = (i + step) % n
                out.append(i)
            return out
        # start[1] から end[0] へ進む側と、start[0] から end[1] へ逆に進む側
        side1 = [self.poly[i] for i in walk(s[1], e[0], +1)]
        side2 = [self.poly[i] for i in walk(s[0], e[1], -1)]
        self.sides = (side1, side2)
        self.centerline = self._centerline(side1, side2)

    @staticmethod
    def _resample(line, n):
        segs = [math.hypot(line[i + 1][0] - line[i][0], line[i + 1][1] - line[i][1]) for i in range(len(line) - 1)]
        total = sum(segs)
        out = []
        for k in range(n):
            d = total * k / (n - 1)
            i = 0
            while i < len(segs) - 1 and d > segs[i]:
                d -= segs[i]
                i += 1
            t = 0.0 if segs[i] == 0 else min(1.0, d / segs[i])
            out.append((line[i][0] + t * (line[i + 1][0] - line[i][0]), line[i][1] + t * (line[i + 1][1] - line[i][1])))
        return out

    def _centerline(self, a, b, n=60):
        ra, rb = self._resample(a, n), self._resample(b, n)
        return [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in zip(ra, rb)]

    def length(self):
        c = self.centerline
        return sum(math.hypot(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]) for i in range(len(c) - 1))

    def widths(self):
        a, b = self._resample(self.sides[0], 60), self._resample(self.sides[1], 60)
        return [math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b)]

    def lane(self, positive=True, frac=0.5):
        """通航帯の中心線（右側通航：進む向きの右の半分の、中心線から frac × 半幅のところ）"""
        a, b = self._resample(self.sides[0], 60), self._resample(self.sides[1], 60)
        c = self.centerline
        out = []
        for i in range(len(c)):
            j = min(i + 1, len(c) - 1)
            k = max(i - 1, 0)
            tx, ty = c[j][0] - c[k][0], c[j][1] - c[k][1]
            tl = math.hypot(tx, ty)
            tx, ty = tx / tl, ty / tl
            # 正の向きの右 = (ty, −tx)
            rx, ry = (ty, -tx) if positive else (-ty, tx)
            half = math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]) / 2
            out.append((c[i][0] + rx * frac * half, c[i][1] + ry * frac * half))
        return out if positive else list(reversed(out))

    def axis_at(self, p):
        """p に最も近い中心線の区間の向き（正の向きの単位ベクトル）と、中心線からの距離"""
        best = (1e18, 0)
        c = self.centerline
        for i in range(len(c) - 1):
            d = dist_to_segment(p, c[i], c[i + 1])
            if d < best[0]:
                best = (d, i)
        i = best[1]
        tx, ty = c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]
        tl = math.hypot(tx, ty)
        return (tx / tl, ty / tl), best[0]


class SeaArea:
    def __init__(self, path):
        self.file_path = path
        self.cfg = json.load(open(path, encoding="utf-8"))
        g = self.cfg["geo_origin"]
        self.origin = GeoOrigin(g["lat_deg"], g["lon_deg"])
        self.refs = {k: self.origin.to_enu(*v) for k, v in self.cfg["references"].items()}
        self.routes = {z["name"]: Route(z, self.refs) for z in self.cfg["zones"] if z["kind"] == "route"}

    def law_point(self, p):
        """{"ref", "bearing_deg", "dist_m"} か {"mid": [点, 点]}（2 点の中点）を ENU に"""
        if "mid" in p:
            a, b = (self.law_point(q) for q in p["mid"])
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return offset(self.refs[p["ref"]], p["bearing_deg"], p["dist_m"])

    def path(self, tr):
        """交通の針路：path_points（法令の点・中点）か path_geo（緯度経度）"""
        if "path_points" in tr:
            return [self.law_point(p) for p in tr["path_points"]]
        return [self.origin.to_enu(*p) for p in tr["path_geo"]]

    def current(self, east, north, t):
        """潮流 [m/s]（東・北）。航路の軸に沿う往復流 peak·sin(2πt/T + φ)、航路の外は exp(−距離 / decay)"""
        c = self.cfg.get("current")
        if not c:
            return (0.0, 0.0)
        r = self.routes[c["zone"]]
        (ax, ay), _ = r.axis_at((east, north))
        d = dist_to_polygon((east, north), r.poly)
        s = c["peak_kt"] * KT * math.sin(2 * math.pi * t / c["period_s"] + math.radians(c["phase_deg"]))
        s *= math.exp(-d / c["outside_decay_m"]) if d > 0 else 1.0
        return (s * ax, s * ay)


def main():
    area = SeaArea(sys.argv[1])
    print(f"{area.cfg['title']}: origin {area.cfg['geo_origin']}")
    for name, r in area.routes.items():
        w = r.widths()
        print(f"  {name}: centerline {r.length():.0f} m, width {min(w):.0f}..{max(w):.0f} m")
        print("   polygon:", " ".join(f"({x:.0f},{y:.0f})" for x, y in r.poly))
        c = r.centerline
        print(f"   start {c[0][0]:.0f},{c[0][1]:.0f} -> end {c[-1][0]:.0f},{c[-1][1]:.0f}")
    for t in (0.0,):
        print("  current at origin:", area.current(0, 0, t))


if __name__ == "__main__":
    main()
