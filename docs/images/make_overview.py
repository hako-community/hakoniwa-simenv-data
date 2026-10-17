"""README の概要図（docs/images/overview.png）を描く。

    python docs/images/make_overview.py

列は「入力 → このリポジトリ → 生成物と使う側」、行は 3 つの機能（環境値・都市と地形・センサー用ジオメトリ）。
箱の文言を変えたらこのファイルを直して描き直す。フォントは Windows の Yu Gothic UI（無ければ Meiryo）。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 720
FONT_CANDIDATES = ["C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"]
BOLD_CANDIDATES = ["C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc"]


def font(cands, size):
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F = font(FONT_CANDIDATES, 17)
FS = font(FONT_CANDIDATES, 14)
FB = font(BOLD_CANDIDATES, 19)

COL = {"in": (30, 300), "tool": (390, 760), "out": (860, 1470)}
PANEL = {"in": ("入力（人が書く・公開データ）", "#eef3fb", "#9fb6d9"),
         "tool": ("hakoniwa-simenv-data", "#fff7df", "#d9c27a"),
         "out": ("生成物 → 使う側", "#eef7ee", "#9ccc9c")}
ROWS = [  # (入力, ツール, [生成物...])
    ("Creatorモデル（JSON）\n基本環境＋局所zone",
     "環境値モデル\ncreator → 実行モデル\nFastSearcher（AABB/BVH）・2D可視化",
     ["area / property / link\n→ 箱庭の環境アセット（PDUで風・温度・気圧・GPS）"]),
    ("PLATEAU CityGML\n建物・TINRelief",
     "city pipeline\n建物 → LOD1 / OBB → MJCF\n地形 → 共通標高格子",
     ["MJCF・HField\n→ Hakoniwa / MuJoCo runtime",
      "GeoTIFF・標高格子\n→ Mapray / Three.js（geo-viewer・web3d-drone）"]),
    ("OBB JSON\n（センサー用の単一真実源）",
     "ペア生成\nobb2mjcf ＋ obb2godot・actors",
     ["env.xml ＋ env.tscn（同じ箱）\n→ mujoco-sensor（mj_ray）・Godot で可視化"]),
]
ROW_Y = [(80, 230), (250, 460), (480, 610)]


def box(d, x0, y0, x1, y1, text, fill="#ffffff", outline="#888888", f=F):
    d.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline=outline, width=2)
    d.multiline_text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=f, fill="#222222", anchor="mm", align="center", spacing=6)


def arrow(d, x0, y0, x1, y1):
    d.line((x0, y0, x1, y1), fill="#555555", width=2)
    d.polygon([(x1, y1), (x1 - 12, y1 - 6), (x1 - 12, y1 + 6)], fill="#555555")


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for k, (x0, x1) in COL.items():
        title, fill, line = PANEL[k]
        d.rounded_rectangle((x0 - 15, 20, x1 + 15, H - 20), radius=18, fill=fill, outline=line, width=2)
        d.text(((x0 + x1) / 2, 48), title, font=FB, fill="#333333", anchor="mm")
    for (src, tool, outs), (y0, y1) in zip(ROWS, ROW_Y):
        ym = (y0 + y1) / 2
        box(d, COL["in"][0], ym - 45, COL["in"][1], ym + 45, src)
        box(d, COL["tool"][0], ym - 55, COL["tool"][1], ym + 55, tool)
        arrow(d, COL["in"][1], ym, COL["tool"][0], ym)
        n = len(outs)
        for i, o in enumerate(outs):
            oy = ym if n == 1 else y0 + 35 + i * ((y1 - y0 - 70) / (n - 1))
            box(d, COL["out"][0], oy - 38, COL["out"][1], oy + 38, o)
            arrow(d, COL["tool"][1], ym, COL["out"][0], oy)
    # 共通の約束（中央の列の下）
    box(d, COL["tool"][0], 630, COL["tool"][1], 690,
        "座標・標高の契約（conventions）\n生成物に入力・範囲・checksum を記録（manifest）", fill="#fdeec4", outline="#d9c27a", f=FS)
    out = Path(__file__).with_name("overview.png")
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
