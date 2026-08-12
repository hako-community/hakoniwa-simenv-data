#!/usr/bin/env python3
"""
OBB JSON (center, half_size, yaw, z) -> Godot scene (.tscn)

Companion to obb2mjcf.py (same directory): emits the SAME boxes
as a Godot scene for visualisation (pattern A). Each building becomes a
StaticBody3D named ``body_<id>`` with a BoxMesh (visual) and a BoxShape3D
collider (so the scene can also serve the Godot-side sensing fallback, pattern B).

Heights are normalised so the global minimum zmin becomes 0 (matches obb2mjcf).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import conventions as conv


def _fmt(x: float) -> str:
    return f"{float(x):.6f}".rstrip("0").rstrip(".") or "0"


def make_tscn(
    items: List[Dict[str, Any]],
    fallback_zmin: float = 0.0,
    fallback_height: float = 5.0,
    root_name: str = "EnvRoot",
    default_rgba=(0.82, 0.82, 0.86, 1.0),
) -> str:
    z_offset = conv.compute_z_offset(items, fallback_zmin)

    sub_blocks: List[str] = []
    node_blocks: List[str] = []
    load_steps = 1  # the scene itself

    for idx, item in enumerate(items):
        box = conv.resolve_box(item, z_offset, fallback_zmin, fallback_height)
        g = conv.obb_to_godot(box)
        gid = g["id"]
        mesh_id = f"BoxMesh_{idx}"
        shape_id = f"BoxShape3D_{idx}"
        mat_id = f"Mat_{idx}"
        rgba = tuple(item.get("rgba", default_rgba))
        sx, sy, sz = g["size"]
        ox, oy, oz = g["origin"]
        (c00, c01, c02), (c10, c11, c12), (c20, c21, c22) = g["basis_cols"]

        sub_blocks.append(
            f'[sub_resource type="BoxMesh" id="{mesh_id}"]\n'
            f"size = Vector3({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})\n"
        )
        sub_blocks.append(
            f'[sub_resource type="BoxShape3D" id="{shape_id}"]\n'
            f"size = Vector3({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})\n"
        )
        sub_blocks.append(
            f'[sub_resource type="StandardMaterial3D" id="{mat_id}"]\n'
            f"albedo_color = Color({_fmt(rgba[0])}, {_fmt(rgba[1])}, "
            f"{_fmt(rgba[2])}, {_fmt(rgba[3])})\n"
        )
        load_steps += 3

        tf = (
            f"Transform3D({_fmt(c00)}, {_fmt(c01)}, {_fmt(c02)}, "
            f"{_fmt(c10)}, {_fmt(c11)}, {_fmt(c12)}, "
            f"{_fmt(c20)}, {_fmt(c21)}, {_fmt(c22)}, "
            f"{_fmt(ox)}, {_fmt(oy)}, {_fmt(oz)})"
        )
        node_blocks.append(
            f'[node name="body_{gid}" type="StaticBody3D" parent="."]\n'
            f"transform = {tf}\n\n"
            f'[node name="MeshInstance3D" type="MeshInstance3D" parent="body_{gid}"]\n'
            f'mesh = SubResource("{mesh_id}")\n'
            f'surface_material_override/0 = SubResource("{mat_id}")\n\n'
            f'[node name="CollisionShape3D" type="CollisionShape3D" parent="body_{gid}"]\n'
            f'shape = SubResource("{shape_id}")\n'
        )

    parts: List[str] = []
    parts.append(f"[gd_scene load_steps={load_steps} format=3]\n")
    parts.extend(sub_blocks)
    parts.append(f'[node name="{root_name}" type="Node3D"]\n')
    parts.extend(node_blocks)
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OBB JSON -> Godot .tscn")
    ap.add_argument("--inp", required=True, help="OBB JSON (key 'results' or 'polygons')")
    ap.add_argument("--out", required=True, help="Output .tscn")
    ap.add_argument("--height", type=float, default=5.0, help="Fallback height")
    ap.add_argument("--zmin", type=float, default=0.0, help="Fallback base z")
    ap.add_argument("--root-name", default="EnvRoot")
    args = ap.parse_args(argv)

    items = conv.load_obb_items(args.inp)
    tscn = make_tscn(items, fallback_zmin=args.zmin, fallback_height=args.height,
                     root_name=args.root_name)
    Path(args.out).write_text(tscn, encoding="utf-8")
    print(f"[INFO] wrote {args.out} ({len(items)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
