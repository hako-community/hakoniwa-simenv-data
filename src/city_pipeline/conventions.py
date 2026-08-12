"""
Coordinate / geometry conventions shared by the envsim-sensor generators.

Single source of truth = OBB JSON (ENU: x=East, y=North, z=Up).
From one OBB item we derive two engine representations that describe the SAME
physical box:

  - MuJoCo (MJCF), matching hakoniwa-envsim/src/city_pipeline/obb2mjcf.py:
        pos   = (cy, -cx, cz)         # MJCF axes: X=North, Y=West(-East), Z=Up
        size  = (sy, sx, half_z)      # half extents (note the sx/sy swap)
        euler = (0, 0, yaw_deg)       # rotation about MJCF Z (=Up)

  - Godot, matching the Godot axis convention used by the drone sim
    (Godot X=East, Y=Up, Z=North; cf. LiDAR PDU conversion x<-Z, y<--X, z<-Y):
        origin = (cx, cz, cy)
        basis cols: col0=(cosθ,0,sinθ), col1=(0,1,0), col2=(-sinθ,0,cosθ)
        BoxMesh.size = (2*sx, zmax-zmin, 2*sy)   # full extents

Both map to the SAME 8 corners in the common ENU frame -- this is what
test_pair_consistency.py verifies.

Heights are normalised so the global minimum zmin becomes 0 (same as obb2mjcf).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------
# OBB JSON loading / normalisation
# --------------------------------------------------------------------------

def load_obb_items(path: str | Path) -> List[Dict[str, Any]]:
    """Load OBB items from JSON (key 'results' or 'polygons')."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("results", data.get("polygons", []))
    if not items:
        raise ValueError(f"No OBB items in {path} (expects key 'results' or 'polygons')")
    return items


def compute_z_offset(items: List[Dict[str, Any]], fallback_zmin: float = 0.0) -> float:
    """Global z offset so that the minimum zmin becomes 0 (mirrors obb2mjcf)."""
    zmins = [float(it["zmin"]) if "zmin" in it else float(fallback_zmin) for it in items]
    return min(zmins) if zmins else 0.0


def resolve_box(
    item: Dict[str, Any],
    z_offset: float = 0.0,
    fallback_zmin: float = 0.0,
    fallback_height: float = 5.0,
) -> Dict[str, Any]:
    """
    Resolve an OBB item to canonical fields after z normalisation.

    Returns dict with: id, cx, cy, sx, sy, yaw_rad, zmin, zmax (z normalised).
    """
    gid = str(item.get("id", "bldg"))
    cx, cy = float(item["center"][0]), float(item["center"][1])
    sx, sy = float(item["half_size"][0]), float(item["half_size"][1])
    yaw = float(item.get("yaw_rad", item.get("yaw", 0.0)))

    if "height" in item:
        height = float(item["height"])
        zmin = float(item.get("zmin", fallback_zmin))
        zmax = zmin + height
    elif "zmin" in item and "zmax" in item:
        zmin = float(item["zmin"])
        zmax = float(item["zmax"])
    else:
        zmin = float(fallback_zmin)
        zmax = zmin + float(fallback_height)

    zmin -= z_offset
    zmax -= z_offset
    return {
        "id": gid, "cx": cx, "cy": cy, "sx": sx, "sy": sy,
        "yaw_rad": yaw, "zmin": zmin, "zmax": zmax,
    }


# --------------------------------------------------------------------------
# Godot transform derivation
# --------------------------------------------------------------------------

def obb_to_godot(box: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolved box (from resolve_box) -> Godot node data.

    Returns: id, origin (3), basis_cols (3x3, columns), size (3 full extents).
    """
    cx, cy = box["cx"], box["cy"]
    sx, sy = box["sx"], box["sy"]
    zmin, zmax = box["zmin"], box["zmax"]
    th = box["yaw_rad"]
    c, s = math.cos(th), math.sin(th)

    cz = 0.5 * (zmin + zmax)
    origin = (cx, cz, cy)                       # (E, Up, N)
    basis_cols = (
        (c, 0.0, s),                            # col0 (mesh X) -> sx axis
        (0.0, 1.0, 0.0),                        # col1 (mesh Y) -> vertical
        (-s, 0.0, c),                           # col2 (mesh Z) -> sy axis
    )
    size = (2.0 * sx, zmax - zmin, 2.0 * sy)
    return {"id": box["id"], "origin": origin, "basis_cols": basis_cols, "size": size}


# --------------------------------------------------------------------------
# Corner computation in the common ENU frame (E, N, Up)
# --------------------------------------------------------------------------

def _sorted_corners(corners: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    return sorted(round3(c) for c in corners)


def round3(v: Tuple[float, float, float], nd: int = 4) -> Tuple[float, float, float]:
    return (round(v[0], nd), round(v[1], nd), round(v[2], nd))


def godot_box_corners_enu(node: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """8 corners of a Godot box, expressed in ENU (E, N, Up)."""
    ox, oy, oz = node["origin"]
    c0, c1, c2 = node["basis_cols"]
    hx, hy, hz = (node["size"][0] / 2.0, node["size"][1] / 2.0, node["size"][2] / 2.0)
    out = []
    for ax in (-hx, hx):
        for ay in (-hy, hy):
            for az in (-hz, hz):
                wx = ox + c0[0] * ax + c1[0] * ay + c2[0] * az
                wy = oy + c0[1] * ax + c1[1] * ay + c2[1] * az
                wz = oz + c0[2] * ax + c1[2] * ay + c2[2] * az
                # Godot (X=E, Y=Up, Z=N) -> ENU (E, N, Up)
                out.append((wx, wz, wy))
    return _sorted_corners(out)


def mjcf_box_corners_enu(pos, size, yaw_deg) -> List[Tuple[float, float, float]]:
    """8 corners of an MJCF box (pos, half-size, euler-z deg) in ENU (E, N, Up)."""
    px, py, pz = pos
    hx, hy, hz = size
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    out = []
    for ax in (-hx, hx):
        for ay in (-hy, hy):
            for az in (-hz, hz):
                # Rz about MJCF Z (Up)
                wx = px + (c * ax - s * ay)
                wy = py + (s * ax + c * ay)
                wz = pz + az
                # MJCF (X=N, Y=-E, Z=Up) -> ENU (E, N, Up)
                out.append((-wy, wx, wz))
    return _sorted_corners(out)


# --------------------------------------------------------------------------
# Parsers (read generated files back for the consistency test)
# --------------------------------------------------------------------------

def parse_mjcf_boxes(xml_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Parse box geoms from an MJCF file. Returns {id: {pos, size, yaw_deg}}."""
    root = ET.parse(str(xml_path)).getroot()
    out: Dict[str, Dict[str, Any]] = {}
    for geom in root.iter("geom"):
        if geom.get("type") != "box":
            continue
        name = geom.get("name", "")
        gid = name[len("geom_"):] if name.startswith("geom_") else name
        pos = tuple(float(x) for x in geom.get("pos", "0 0 0").split())
        size = tuple(float(x) for x in geom.get("size", "0 0 0").split())
        euler = geom.get("euler", "0 0 0").split()
        yaw_deg = float(euler[2]) if len(euler) >= 3 else 0.0
        out[gid] = {"pos": pos, "size": size, "yaw_deg": yaw_deg}
    return out


def parse_tscn_boxes(tscn_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse box bodies from a generated .tscn. Returns
    {id: {origin, basis_cols, size}}.

    Relies on the layout emitted by obb2godot (BoxMesh sub_resources keyed by id,
    StaticBody3D nodes named body_<id> with a transform).
    """
    text = Path(tscn_path).read_text(encoding="utf-8")
    lines = text.splitlines()

    # sub_resource BoxMesh sizes keyed by resource id
    mesh_size: Dict[str, Tuple[float, float, float]] = {}
    cur_res = None
    cur_type = None
    for ln in lines:
        st = ln.strip()
        if st.startswith("[sub_resource"):
            cur_type = _attr(st, "type")
            cur_res = _attr(st, "id")
        elif st.startswith("[node") or st.startswith("[gd_scene"):
            cur_res = None
            cur_type = None
        elif cur_res and cur_type == "BoxMesh" and st.startswith("size = Vector3"):
            mesh_size[cur_res] = _vec3(st)

    out: Dict[str, Dict[str, Any]] = {}
    cur_id = None
    cur_origin = None
    cur_cols = None
    cur_mesh_ref = None
    in_body = False

    def flush():
        if cur_id and cur_origin and cur_cols and cur_mesh_ref in mesh_size:
            out[cur_id] = {
                "origin": cur_origin,
                "basis_cols": cur_cols,
                "size": mesh_size[cur_mesh_ref],
            }

    for ln in lines:
        st = ln.strip()
        if st.startswith("[node"):
            ntype = _attr(st, "type")
            nname = _attr(st, "name")
            if ntype == "StaticBody3D" and nname and nname.startswith("body_"):
                flush()
                cur_id = nname[len("body_"):]
                cur_origin = None
                cur_cols = None
                cur_mesh_ref = None
                in_body = True
            elif ntype == "StaticBody3D":
                flush()
                in_body = False
                cur_id = None
        elif in_body and st.startswith("transform = Transform3D"):
            f = _floats(st)
            if len(f) == 12:
                cur_cols = ((f[0], f[1], f[2]), (f[3], f[4], f[5]), (f[6], f[7], f[8]))
                cur_origin = (f[9], f[10], f[11])
        elif in_body and st.startswith("mesh = SubResource"):
            cur_mesh_ref = _quoted(st)
    flush()
    return out


# --- tiny text helpers (avoid heavy deps) ---

def _attr(line: str, key: str) -> str | None:
    token = f'{key}="'
    i = line.find(token)
    if i < 0:
        return None
    i += len(token)
    j = line.find('"', i)
    return line[i:j] if j > i else None


def _quoted(line: str) -> str | None:
    i = line.find('"')
    if i < 0:
        return None
    j = line.find('"', i + 1)
    return line[i + 1:j] if j > i else None


def _vec3(line: str) -> Tuple[float, float, float]:
    f = _floats(line)
    return (f[0], f[1], f[2])


def _floats(line: str) -> List[float]:
    i = line.find("(")
    j = line.rfind(")")
    inner = line[i + 1:j]
    return [float(x.strip()) for x in inner.split(",") if x.strip() != ""]
