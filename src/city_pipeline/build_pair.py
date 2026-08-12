#!/usr/bin/env python3
"""
build_pair: OBB JSON -> (env.xml for MuJoCo sensing) + (env.tscn for Godot view)

Single source of truth = OBB JSON. The MuJoCo XML is produced by obb2mjcf.py and
the Godot scene by obb2godot.py -- both siblings of this file. Both describe the
same boxes.

obb2mjcf is located via (in order): --obb2mjcf, this same directory, or
$HAKONIWA_ENVSIM_DIR (kept for compatibility with the pre-consolidation layout).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import obb2godot


def _find_obb2mjcf(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    # obb2mjcf.py now lives next to this file (hakoniwa-simenv-data/src/city_pipeline).
    candidates.append(Path(__file__).resolve().parent / "obb2mjcf.py")
    env_dir = os.environ.get("HAKONIWA_ENVSIM_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "src" / "city_pipeline" / "obb2mjcf.py")
    for c in candidates:
        if c and c.is_file():
            return c
    raise FileNotFoundError(
        "obb2mjcf.py not found. Pass --obb2mjcf or set HAKONIWA_ENVSIM_DIR.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def build_pair(inp: str, out_xml: str, out_tscn: str,
               obb2mjcf_path: str | None = None,
               height: float = 5.0, zmin: float = 0.0,
               floor: bool = False) -> None:
    mjcf_tool = _find_obb2mjcf(obb2mjcf_path)

    cmd = [sys.executable, str(mjcf_tool), "--inp", inp, "--out", out_xml,
           "--height", str(height), "--zmin", str(zmin)]
    if floor:
        cmd.append("--floor")
    print(f"[INFO] obb2mjcf: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    obb2godot.main(["--inp", inp, "--out", out_tscn,
                    "--height", str(height), "--zmin", str(zmin)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OBB JSON -> env.xml + env.tscn")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out-xml", required=True)
    ap.add_argument("--out-tscn", required=True)
    ap.add_argument("--obb2mjcf", default=None, help="Path to envsim obb2mjcf.py")
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--floor", action="store_true")
    args = ap.parse_args(argv)
    build_pair(args.inp, args.out_xml, args.out_tscn, args.obb2mjcf,
               args.height, args.zmin, args.floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
