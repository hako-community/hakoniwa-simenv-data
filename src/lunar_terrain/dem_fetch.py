"""dem_fetch.py — dem.lock.json の DEM を deps/dem/ に取り、SHA256 を照合する。

    python src/lunar_terrain/dem_fetch.py            # 全部（既にあって一致するものは飛ばす）
    python src/lunar_terrain/dem_fetch.py --only Site01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = HERE / "dem.lock.json"
DEPS = ROOT / "deps" / "dem"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="path に含まれる文字列で絞る")
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for e in lock["files"]:
        if a.only and a.only not in e["path"]:
            continue
        dst = DEPS / e["path"]
        if dst.exists() and sha256(dst) == e["sha256"]:
            print(f"[dem_fetch] ok   {dst.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[dem_fetch] get  {e['url']}")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(e["url"], tmp)
        if sha256(tmp) != e["sha256"]:
            tmp.unlink()
            raise SystemExit(f"SHA256 が違います: {e['url']}")
        tmp.replace(dst)
        print(f"[dem_fetch] done {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
