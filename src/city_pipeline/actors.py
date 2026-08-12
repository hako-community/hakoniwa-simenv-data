"""Add dynamic ACTOR bodies to a generated sensor env.xml (A-1).

What this is for:
    env.xml itself is produced by obb2mjcf.py. Actors are a *sensing* concern --
    they exist so that hakoniwa-mujoco-sensor can inject a moving object (another
    drone, a vehicle) into the kinematic sensing world -- so this runs as a
    post-processing pass over the generated env.xml.

Why a free joint and not a mocap body:
    MuJoCo returns zero velocity from mj_objectVelocity for a mocap body, so the
    radar's Doppler reading would stay 0 -- which defeats the whole purpose. A
    body with a free joint lets the runtime write both qpos AND qvel, so the
    ray-cast backend reports a real relative velocity.

Radar cross-section (--rcs NAME=M2):
    Writes an RCS in m^2 into the geom's first user slot, which the MuJoCo ray
    caster reads back and the radar uses to scale its detection range by
    (sigma/sigma_ref)^(1/4).

    IMPORTANT -- what to put here. The radar model ALREADY accounts for a
    target's apparent size: a bigger box intercepts proportionally more rays, so
    it is detected further out without any annotation. This attribute is the
    part that geometry cannot express -- how reflective the surface is compared
    with a same-sized reference target. Setting sigma proportional to a target's
    physical area therefore DOUBLE-COUNTS its size. Use it for material and
    shape differences: a corner reflector or a metal airframe above the
    reference, a foam or radar-absorbent surface below it.

Usage:
    PYTHONPATH=src/city_pipeline python3 -m actors --xml generated/env.xml \
        --out generated/env_actors.xml --actor drone2 --actor obstacle:0.4,0.4,0.4

    # a metallic airframe reflects far better than the 1 m^2 reference
    PYTHONPATH=src/city_pipeline python3 -m actors --xml generated/env.xml \
        --out generated/env_actors_rcs.xml --actor drone2 --rcs drone2=4.0

    # display-only ground: drawn by Godot, absent from the sensing world
    PYTHONPATH=src/city_pipeline python3 -m actors --xml generated/env.xml \
        --out generated/env_actors_nofloor.xml --actor drone2 --drop ground
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_HALF_SIZE = (0.25, 0.25, 0.15)   # m, a small quadrotor-ish box
DEFAULT_RGBA = "0.9 0.3 0.2 1"
FAR_AWAY_Z = -50.0                       # parked below the floor until injected


def _parse_actor(spec: str) -> tuple[str, tuple[float, float, float]]:
    """`name` or `name:hx,hy,hz`"""
    if ":" not in spec:
        return spec, DEFAULT_HALF_SIZE
    name, dims = spec.split(":", 1)
    parts = [float(v) for v in dims.split(",")]
    if len(parts) != 3:
        raise ValueError(f"actor size must be hx,hy,hz: {spec}")
    return name, (parts[0], parts[1], parts[2])


def _parse_rcs(specs: list[str]) -> dict[str, float]:
    """`NAME=VALUE` pairs -> {name: m^2}. NAME is an actor name or an OBB id."""
    out: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--rcs must be NAME=VALUE (m^2): {spec}")
        name, value = spec.split("=", 1)
        rcs = float(value)
        if rcs <= 0.0:
            raise ValueError(f"--rcs value must be > 0 (m^2): {spec}")
        out[name.strip()] = rcs
    return out


def _ensure_nuser_geom(root: ET.Element) -> None:
    """MuJoCo only allocates geom user slots if <size nuser_geom="..."> asks for it.

    Without this the RCS written below is silently dropped at model-compile time.
    """
    size = root.find("size")
    if size is None:
        size = ET.Element("size")
        root.insert(0, size)
    if int(size.get("nuser_geom", "0") or 0) < 1:
        size.set("nuser_geom", "1")


def _apply_rcs(world: ET.Element, rcs_map: dict[str, float]) -> list[str]:
    """Write RCS into each named body's geoms. Returns the names that matched."""
    applied = []
    for name, rcs in rcs_map.items():
        # Accept the bare OBB id, the body_<id> form, or an actor name.
        candidates = {name, f"body_{name}", f"actor_{name}"}
        for body in world.findall("body"):
            if body.get("name") not in candidates:
                continue
            for geom in body.findall("geom"):
                geom.set("user", f"{rcs:g}")
            applied.append(name)
    return applied


def add_actors(xml_in: str | Path, xml_out: str | Path, actors: list[str],
               drop: list[str] | None = None,
               rcs: list[str] | None = None) -> Path:
    tree = ET.parse(str(xml_in))
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{xml_in}: no <worldbody>")

    # Display-only geometry: a body can exist in the OBB (so Godot draws it) yet be
    # deliberately absent from the SENSING world. The ground slab of an open-field
    # scene is the case in point -- keeping it would flood the radar with static
    # clutter and bury the one object the scenario is about (the other aircraft).
    for spec in (drop or []):
        name = spec if spec.startswith("body_") else f"body_{spec}"
        for b in list(world.findall("body")):
            if b.get("name") == name:
                world.remove(b)

    existing = {b.get("name") for b in world.findall("body")}
    for spec in actors:
        name, half = _parse_actor(spec)
        body_name = f"actor_{name}"
        if body_name in existing:
            continue
        # Parked far below the floor: until the runtime calls SetActor() the actor
        # must not be visible to any sensor.
        body = ET.SubElement(world, "body", {
            "name": body_name,
            "pos": f"0 0 {FAR_AWAY_Z}",
        })
        ET.SubElement(body, "freejoint", {"name": f"{body_name}_free"})
        ET.SubElement(body, "geom", {
            "name": f"geom_{body_name}",
            "type": "box",
            "size": f"{half[0]} {half[1]} {half[2]}",
            "rgba": DEFAULT_RGBA,
            # same contact flags the walls use: visible to mj_ray, no dynamics needed
            "contype": "1",
            "conaffinity": "0",
        })

    # Radar cross-section per target. Only touches the XML when asked, so
    # unannotated scenes stay byte-identical to what they were before.
    rcs_map = _parse_rcs(rcs or [])
    if rcs_map:
        _ensure_nuser_geom(root)
        applied = _apply_rcs(world, rcs_map)
        missing = sorted(set(rcs_map) - set(applied))
        if missing:
            raise ValueError(
                f"--rcs named bodies that are not in {xml_in}: {', '.join(missing)}")

    out = Path(xml_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="utf-8", xml_declaration=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Append free-joint actor bodies to a sensor env.xml")
    ap.add_argument("--xml", required=True, help="input env.xml (from obb2mjcf)")
    ap.add_argument("--out", required=True, help="output env.xml with actors")
    ap.add_argument("--actor", action="append", default=[],
                    help="actor spec: NAME or NAME:hx,hy,hz (repeatable)")
    ap.add_argument("--drop", action="append", default=[],
                    help="OBB id (or body_<id>) to REMOVE from the sensing world, "
                         "for display-only geometry such as a ground slab (repeatable)")
    ap.add_argument("--rcs", action="append", default=[], metavar="NAME=M2",
                    help="radar cross-section in m^2 for a target: NAME is an actor "
                         "name or an OBB id (repeatable). Expresses reflectivity "
                         "differences; see the module docstring on double-counting")
    args = ap.parse_args(argv)
    if not args.actor:
        ap.error("at least one --actor is required")
    out = add_actors(args.xml, args.out, args.actor, args.drop, args.rcs)
    print(f"[INFO] wrote {out} with actors: {', '.join(args.actor)}"
          + (f" (dropped: {', '.join(args.drop)})" if args.drop else "")
          + (f" (rcs: {', '.join(args.rcs)})" if args.rcs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
