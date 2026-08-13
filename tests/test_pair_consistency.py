"""
M1 consistency test: env.xml (MuJoCo) and env.tscn (Godot) generated from one
OBB JSON must describe the SAME boxes.

For every box id, the 8 corners computed from the MuJoCo geom and from the Godot
node are compared in a common ENU frame. This validates position, size, axis
mapping and yaw handling between the two generators.

Run: python3 tests/test_pair_consistency.py
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "city_pipeline"))

import build_pair  # noqa: E402
import conventions as conv  # noqa: E402

OBB = REPO / "examples" / "sensor_envs" / "simple_room" / "simple_room.obb.json"
TOL = 1e-3


def _max_corner_diff(a, b):
    return max(
        max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]), abs(pa[2] - pb[2]))
        for pa, pb in zip(a, b)
    )


def _yaw_case(failures):
    """A rotated (yaw!=0) box must also stay consistent between the two engines."""
    import json
    with tempfile.TemporaryDirectory() as td:
        obb = Path(td) / "rot.obb.json"
        obb.write_text(json.dumps({"coordinate_system": "enu", "results": [
            {"id": "rot_box", "center": [2.0, -1.0], "half_size": [1.5, 0.5],
             "yaw_rad": 0.5236, "zmin": 0.0, "zmax": 4.0}
        ]}), encoding="utf-8")
        out_xml = Path(td) / "env.xml"
        out_tscn = Path(td) / "env.tscn"
        build_pair.build_pair(str(obb), str(out_xml), str(out_tscn))
        mj = conv.parse_mjcf_boxes(out_xml)["rot_box"]
        gd = conv.parse_tscn_boxes(out_tscn)["rot_box"]
        diff = _max_corner_diff(
            conv.mjcf_box_corners_enu(mj["pos"], mj["size"], mj["yaw_deg"]),
            conv.godot_box_corners_enu(gd),
        )
        _check(failures, diff < TOL, f"[rot_box yaw=30deg] corners match (max diff={diff:.6f}m)")


def main() -> int:
    failures = []
    n_items = len(conv.load_obb_items(OBB))

    with tempfile.TemporaryDirectory() as td:
        out_xml = Path(td) / "env.xml"
        out_tscn = Path(td) / "env.tscn"
        build_pair.build_pair(str(OBB), str(out_xml), str(out_tscn))

        mj = conv.parse_mjcf_boxes(out_xml)
        gd = conv.parse_tscn_boxes(out_tscn)

        # 1) counts
        _check(failures, len(mj) == n_items, f"MJCF box count {len(mj)} == {n_items}")
        _check(failures, len(gd) == n_items, f"Godot box count {len(gd)} == {n_items}")
        _check(failures, set(mj) == set(gd), f"ids match (mjcf={sorted(mj)} godot={sorted(gd)})")

        # 2) per-id corner consistency in ENU
        for gid in sorted(set(mj) & set(gd)):
            mc = conv.mjcf_box_corners_enu(mj[gid]["pos"], mj[gid]["size"], mj[gid]["yaw_deg"])
            gc = conv.godot_box_corners_enu(gd[gid])
            diff = _max_corner_diff(mc, gc)
            _check(failures, diff < TOL, f"[{gid}] corners match (max diff={diff:.6f}m)")

    _yaw_case(failures)

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
        return 1
    print("RESULT: PASS")
    return 0


def _check(failures, cond, msg):
    if cond:
        print(f"  [ ok ] {msg}")
    else:
        failures.append(msg)
        print(f"  [FAIL] {msg}")


if __name__ == "__main__":
    raise SystemExit(main())
