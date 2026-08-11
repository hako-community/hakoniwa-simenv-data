#!/usr/bin/env python3
"""Windows-native PLATEAU CityGML to MuJoCo pipeline API.

The functions in this module compose the existing city_pipeline modules in
process.  No bash, WSL, Docker, or child Python process is required.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

import gml2obb
import gml_extract
import gml_indexer
import gml_lod1_extract
import obb2mjcf


BUILDING_BODY_PREFIX = "body_bldg_"
BUILDING_GEOM_PREFIX = "geom_bldg_"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _epsg_number(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).upper()
    if text.startswith("EPSG:"):
        text = text.split(":", 1)[1]
    return int(text)


def load_scenario(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("name", "origin", "horizontal_crs", "extent", "geometry", "collision")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"scenario is missing required keys: {', '.join(missing)}")

    origin = data["origin"]
    for name in ("latitude", "longitude", "altitude"):
        if name not in origin:
            raise ValueError(f"scenario origin is missing '{name}'")

    extent = data["extent"]
    for name in ("north_south_half_m", "east_west_half_m"):
        if float(extent.get(name, 0)) <= 0:
            raise ValueError(f"scenario extent '{name}' must be positive")

    mode = data["collision"].get("mode")
    if mode not in {"all", "drone", "none"}:
        raise ValueError("scenario collision.mode must be all, drone, or none")
    return data


def filter_citygml(source_root: Path, output_root: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    if not (source_root / "udx").is_dir():
        raise ValueError(f"CityGML root must contain an 'udx' directory: {source_root}")

    if output_root.exists():
        marker = output_root / "query_meta.json"
        if output_root.name != "citygml-extracted" or not marker.is_file():
            raise ValueError(
                "refusing to replace an unmanaged output directory; "
                f"expected a generated citygml-extracted directory: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    origin = scenario["origin"]
    extent = scenario["extent"]
    query_bbox = gml_extract.make_query_bbox(
        float(origin["latitude"]),
        float(origin["longitude"]),
        float(extent["north_south_half_m"]),
        float(extent["east_west_half_m"]),
    )
    kinds = set(scenario.get("citygml", {}).get("kinds", ["bldg"]))
    records = gml_indexer.build_index(source_root)
    selected: list[dict[str, str]] = []

    for directory in ("codelists", "metadata", "schemas", "specification"):
        source = source_root / directory
        if source.exists():
            shutil.copytree(source, output_root / directory)
    for filename in ("README.md", "13113_indexmap_op.pdf"):
        source = source_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)

    for item in records:
        if item["kind"] not in kinds:
            continue
        if not gml_extract.is_intersect(query_bbox, gml_extract.mesh_to_bbox(item["mesh"])):
            continue
        source_file = Path(item["path"])
        relative = source_file.relative_to(source_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)

        prefix = f"{item['mesh']}_{item['kind']}"
        for sibling in source_file.parent.iterdir():
            if sibling == source_file or not sibling.name.startswith(prefix):
                continue
            sibling_destination = output_root / sibling.relative_to(source_root)
            if sibling.is_dir():
                shutil.copytree(sibling, sibling_destination, dirs_exist_ok=True)
            else:
                sibling_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sibling, sibling_destination)
        selected.append({
            "mesh": item["mesh"],
            "kind": item["kind"],
            "path": relative.as_posix(),
        })

    if not selected:
        raise ValueError("no CityGML files intersect the configured extent")

    query_meta = {
        "center_lat": float(origin["latitude"]),
        "center_lon": float(origin["longitude"]),
        "ns_m": float(extent["north_south_half_m"]),
        "ew_m": float(extent["east_west_half_m"]),
        "bbox": query_bbox,
    }
    _write_json(output_root / "query_meta.json", query_meta)
    return {"indexed_count": len(records), "selected": selected, "query": query_meta}


def extract_lod1(
    extracted_root: Path,
    output_path: Path,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    city_options = scenario.get("citygml", {})
    source_epsg = _epsg_number(scenario.get("source_crs", "EPSG:4326"))
    target_epsg = _epsg_number(scenario["horizontal_crs"])
    swap_latlon = bool(city_options.get("swap_latlon", True))
    base_epsilon = float(city_options.get("base_epsilon_m", 0.2))
    pattern = city_options.get("pattern", "*bldg*_op.gml")
    origin = scenario["origin"]
    extent = scenario["extent"]

    origin_points = [(float(origin["longitude"]), float(origin["latitude"]), 0.0)]
    if not swap_latlon:
        origin_points = [(float(origin["latitude"]), float(origin["longitude"]), 0.0)]
    projected_origin = gml_lod1_extract.project_xy(
        origin_points,
        src_epsg=source_epsg,
        dst_epsg=target_epsg,
    )[0]
    origin_xy = (projected_origin[0], projected_origin[1])
    bounds = {
        "ns_m": float(extent["north_south_half_m"]),
        "ew_m": float(extent["east_west_half_m"]),
    }

    paths = gml_lod1_extract.collect_gml_paths(extracted_root, pattern)
    if not paths:
        raise ValueError(f"no GML found for pattern '{pattern}' under {extracted_root}")

    polygons: list[dict[str, Any]] = []
    for gml_path in paths:
        records = gml_lod1_extract.extract_buildings_lod1(
            gml_path=gml_path,
            to_epsg=target_epsg,
            src_epsg=source_epsg,
            base_eps=base_epsilon,
            swap_latlon=swap_latlon,
            origin_xy=origin_xy,
            bounds=bounds,
        )
        source_name = gml_path.relative_to(extracted_root).as_posix()
        for record in records:
            item = dict(record)
            item["source_gml"] = source_name
            polygons.append(item)

    data = {
        "version": "1.0",
        "crs": f"EPSG:{target_epsg}",
        "coordinate_system": "relative-enu",
        "origin": {
            "lat": float(origin["latitude"]),
            "lon": float(origin["longitude"]),
            "altitude": float(origin["altitude"]),
            "x": float(origin_xy[0]),
            "y": float(origin_xy[1]),
        },
        "bounds": bounds,
        "polygons": polygons,
    }
    _write_json(output_path, data)
    return data


def convert_lod1_to_obb(lod1_data: dict[str, Any], output_path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    geometry = scenario["geometry"]
    waste_threshold = geometry.get("waste_threshold")
    if waste_threshold is not None:
        waste_threshold = float(waste_threshold)
    wall_thickness = float(geometry.get("wall_thickness_m", 0.1))
    results: list[dict[str, Any]] = []

    for polygon in lod1_data["polygons"]:
        points = np.array(polygon["vertices"], dtype=float)
        if len(points) < 3:
            continue
        polygon_id = str(polygon.get("id", "poly"))
        area_polygon = gml2obb.polygon_area(points)
        center, half_size, yaw, rectangle, area_obb = gml2obb.min_area_rect_calipers(points)
        waste_ratio = area_obb / area_polygon if area_polygon > 0 else None
        use_walls = (
            waste_ratio is not None
            and waste_threshold is not None
            and waste_ratio > waste_threshold
        )
        common = (
            polygon.get("zmin"),
            polygon.get("zmax"),
            polygon.get("height"),
            polygon.get("source_gml"),
        )
        if use_walls:
            wall_records = gml2obb.make_wall_records(
                polygon_id,
                points,
                *common,
                wall_thickness=wall_thickness,
                waste_ratio=waste_ratio,
            )
            if wall_records:
                results.extend(wall_records)
                continue
        results.append(gml2obb.make_obb_record(
            polygon_id,
            center,
            half_size,
            yaw,
            rectangle,
            area_obb,
            *common,
            waste_ratio=waste_ratio,
        ))

    data = {
        "version": "1.0",
        "mode": "after",
        "crs": lod1_data["crs"],
        "coordinate_system": lod1_data["coordinate_system"],
        "origin": lod1_data["origin"],
        "bounds": lod1_data["bounds"],
        "results": results,
    }
    _write_json(output_path, data)
    return data


def generate_buildings_mjcf(
    obb_data: dict[str, Any],
    output_path: Path,
    scenario: dict[str, Any],
    collision_mode: str | None = None,
) -> dict[str, int]:
    items = copy.deepcopy(obb_data["results"])
    # CityGML elevations are absolute heights.  Keep the generated MuJoCo
    # world compatible with the legacy converter by rebasing the lowest
    # building footprint to z=0 before applying the scenario offset.
    zmin_values = [float(item.get("zmin", 0.0)) for item in items]
    z_baseline = min(zmin_values, default=0.0)
    z_offset = float(scenario.get("vertical", {}).get("z_offset", 0.0))
    for item in items:
        if "zmin" in item:
            item["zmin"] = float(item["zmin"]) - z_baseline + z_offset
        if "zmax" in item:
            item["zmax"] = float(item["zmax"]) - z_baseline + z_offset

    axes = scenario.get("mujoco_axes", {"x": "north", "y": "-east", "z": "up"})
    if axes != {"x": "north", "y": "-east", "z": "up"}:
        raise ValueError(f"unsupported mujoco_axes: {axes}")

    def pos_fn(east: float, north: float, up: float) -> tuple[float, float, float]:
        return north, -east, up

    # MuJoCo's default compiler angle unit is degrees.  Swapping the X/Y half
    # sizes is equivalent to applying the required -90 degree ENU axis change
    # for a box, up to the box's 180 degree symmetry.
    def yaw_fn(yaw_enu: float) -> float:
        return math.degrees(yaw_enu)

    def size_fn(size_east: float, size_north: float) -> tuple[float, float]:
        return size_north, size_east

    mode = collision_mode or scenario["collision"]["mode"]
    root = obb2mjcf.make_mjcf(
        items=items,
        model_name=f"{scenario['name']}_buildings",
        collide_mode=mode,
        pos_fn=pos_fn,
        yaw_fn=yaw_fn,
        sxy_fn=size_fn,
    )
    obb2mjcf.indent(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return {
        "building_record_count": len(items),
        "building_body_count": len(root.findall("./worldbody/body")),
        "building_geom_count": len(root.findall("./worldbody/body/geom")),
    }


def integrate_mjcf(base_path: Path, buildings_path: Path, base_output: Path, integrated_output: Path) -> dict[str, int]:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    base_tree = ET.parse(base_path, parser=parser)
    base_root = base_tree.getroot()
    worldbody = base_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"base MJCF has no worldbody: {base_path}")

    removed = 0
    for child in list(worldbody):
        name = child.get("name", "")
        if name.startswith(BUILDING_BODY_PREFIX) or name.startswith(BUILDING_GEOM_PREFIX):
            worldbody.remove(child)
            removed += 1

    obb2mjcf.indent(base_root)
    base_output.parent.mkdir(parents=True, exist_ok=True)
    base_output.write_text(ET.tostring(base_root, encoding="unicode") + "\n", encoding="utf-8")

    buildings_root = ET.parse(buildings_path).getroot()
    buildings_worldbody = buildings_root.find("worldbody")
    if buildings_worldbody is None:
        raise ValueError(f"buildings MJCF has no worldbody: {buildings_path}")
    added = 0
    for child in list(buildings_worldbody):
        worldbody.append(copy.deepcopy(child))
        added += 1

    obb2mjcf.indent(base_root)
    integrated_output.write_text(ET.tostring(base_root, encoding="unicode") + "\n", encoding="utf-8")
    return {"removed_existing_building_nodes": removed, "added_building_nodes": added}


def validate_mjcf(path: Path) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    building_bodies = 0
    building_geoms = 0
    for index in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        building_bodies += int(name.startswith(BUILDING_BODY_PREFIX))
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        building_geoms += int(name.startswith(BUILDING_GEOM_PREFIX))
    return {
        "mujoco_version": mujoco.__version__,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "building_body_count": building_bodies,
        "building_geom_count": building_geoms,
    }


def compare_building_positions(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    def positions(path: Path) -> dict[str, tuple[float, float, float]]:
        root = ET.parse(path).getroot()
        result: dict[str, tuple[float, float, float]] = {}
        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            if not name.startswith(BUILDING_GEOM_PREFIX) or not geom.get("pos"):
                continue
            values = tuple(float(value) for value in geom.get("pos", "").split())
            if len(values) == 3:
                result[name] = values
        return result

    reference = positions(reference_path)
    candidate = positions(candidate_path)
    common = sorted(set(reference) & set(candidate))
    if not common:
        return {"matched_geom_count": 0}
    horizontal = [
        math.hypot(reference[name][0] - candidate[name][0], reference[name][1] - candidate[name][1])
        for name in common
    ]
    vertical = [abs(reference[name][2] - candidate[name][2]) for name in common]
    spatial = [
        math.sqrt(horizontal[index] ** 2 + vertical[index] ** 2)
        for index in range(len(common))
    ]
    return {
        "matched_geom_count": len(common),
        "horizontal_rms_m": math.sqrt(sum(value * value for value in horizontal) / len(horizontal)),
        "horizontal_mean_m": sum(horizontal) / len(horizontal),
        "horizontal_max_m": max(horizontal),
        "vertical_rms_m": math.sqrt(sum(value * value for value in vertical) / len(vertical)),
        "vertical_mean_m": sum(vertical) / len(vertical),
        "vertical_max_m": max(vertical),
        "spatial_rms_m": math.sqrt(sum(value * value for value in spatial) / len(spatial)),
        "spatial_max_m": max(spatial),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
