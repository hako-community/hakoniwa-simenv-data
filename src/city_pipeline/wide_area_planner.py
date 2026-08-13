"""Read-only planning helpers for wide-area PLATEAU / Mapray datasets."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import gml_indexer
from gml_extract import is_intersect, make_query_bbox, mesh_to_bbox


EARTH_RADIUS_M = 6_378_137.0


def _require_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _resolve_workspace_path(workspace_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _iter_positions(coordinates: Any):
    if not isinstance(coordinates, list):
        return
    if len(coordinates) >= 2 and all(isinstance(value, (int, float)) for value in coordinates[:2]):
        yield float(coordinates[0]), float(coordinates[1])
        return
    for child in coordinates:
        yield from _iter_positions(child)


def _feature_bbox(feature: dict[str, Any]) -> dict[str, float]:
    positions = list(_iter_positions(feature.get("geometry", {}).get("coordinates")))
    if not positions:
        raise ValueError("AOI feature contains no coordinates")
    longitudes = [position[0] for position in positions]
    latitudes = [position[1] for position in positions]
    return {
        "min_lat": min(latitudes),
        "max_lat": max(latitudes),
        "min_lon": min(longitudes),
        "max_lon": max(longitudes),
    }


def _bbox_dimensions_m(bbox: dict[str, float]) -> tuple[float, float]:
    center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    height = math.radians(bbox["max_lat"] - bbox["min_lat"]) * EARTH_RADIUS_M
    width = (
        math.radians(bbox["max_lon"] - bbox["min_lon"])
        * EARTH_RADIUS_M
        * math.cos(math.radians(center_lat))
    )
    return width, height


def _buffer_bbox(bbox: dict[str, float], buffer_m: float) -> dict[str, float]:
    center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2.0
    expansion = make_query_bbox(center_lat, center_lon, buffer_m, buffer_m)
    return {
        "min_lat": bbox["min_lat"] - (center_lat - expansion["min_lat"]),
        "max_lat": bbox["max_lat"] + (expansion["max_lat"] - center_lat),
        "min_lon": bbox["min_lon"] - (center_lon - expansion["min_lon"]),
        "max_lon": bbox["max_lon"] + (expansion["max_lon"] - center_lon),
    }


def _associated_assets(gml_path: Path, mesh: str, kind: str) -> list[Path]:
    prefix = f"{mesh}_{kind}"
    assets = [gml_path]
    for sibling in gml_path.parent.iterdir():
        if sibling == gml_path or not sibling.name.startswith(prefix):
            continue
        if sibling.is_file():
            assets.append(sibling)
        elif sibling.is_dir():
            assets.extend(path for path in sibling.rglob("*") if path.is_file())
    return assets


def _inspect_source(
    source: dict[str, Any], workspace_root: Path, query_bbox: dict[str, float]
) -> dict[str, Any]:
    source_id = str(source.get("id", "")).strip()
    root_value = str(source.get("root", "")).strip()
    if not source_id or not root_value:
        raise ValueError("every source requires id and root")
    root = _resolve_workspace_path(workspace_root, root_value)
    if not root.is_dir():
        raise ValueError(f"PLATEAU source is missing: {root}")

    required_kinds = tuple(str(value) for value in source.get("required_kinds", ["bldg", "dem"]))
    selected = []
    unique_assets: set[Path] = set()
    for record in gml_indexer.build_index(root):
        if record["kind"] not in required_kinds:
            continue
        if not is_intersect(query_bbox, mesh_to_bbox(record["mesh"])):
            continue
        path = Path(record["path"])
        assets = _associated_assets(path, record["mesh"], record["kind"])
        unique_assets.update(assets)
        selected.append(
            {
                "mesh": record["mesh"],
                "kind": record["kind"],
                "path": path.relative_to(root).as_posix(),
                "gml_bytes": path.stat().st_size,
                "asset_file_count": len(assets),
                "asset_bytes": sum(asset.stat().st_size for asset in assets),
            }
        )

    kind_counts = Counter(item["kind"] for item in selected)
    missing_kinds = [kind for kind in required_kinds if kind_counts[kind] == 0]
    coverage = str(source.get("aoi_coverage", "unknown"))
    return {
        "id": source_id,
        "root": root_value,
        "municipality_code": source.get("municipality_code"),
        "coverage_scope": source.get("coverage_scope", "unknown"),
        "aoi_coverage": coverage,
        "selected_gml_count": len(selected),
        "selected_asset_file_count": len(unique_assets),
        "selected_asset_bytes": sum(path.stat().st_size for path in unique_assets),
        "counts_by_kind": dict(sorted(kind_counts.items())),
        "missing_required_kinds": missing_kinds,
        "records": sorted(selected, key=lambda item: (item["kind"], item["mesh"], item["path"])),
    }


def build_wide_area_plan(config_path: Path, workspace_root: Path) -> dict[str, Any]:
    """Build a deterministic, read-only inventory and readiness report."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported wide-area config schema")
    scenario_id = str(config.get("scenario_id", "")).strip()
    if not scenario_id:
        raise ValueError("scenario_id is required")

    aoi = config.get("aoi", {})
    operations_value = str(aoi.get("operations_geojson", "")).strip()
    feature_id = str(aoi.get("feature_id", "")).strip()
    if not operations_value or not feature_id:
        raise ValueError("aoi.operations_geojson and aoi.feature_id are required")
    operations_path = _resolve_workspace_path(workspace_root, operations_value)
    operations = json.loads(operations_path.read_text(encoding="utf-8"))
    feature = next(
        (
            item
            for item in operations.get("features", [])
            if item.get("properties", {}).get("id") == feature_id
        ),
        None,
    )
    if feature is None:
        raise ValueError(f"AOI feature not found: {feature_id}")

    bbox = _feature_bbox(feature)
    width_m, height_m = _bbox_dimensions_m(bbox)
    nominal_width = _require_number(aoi.get("nominal_width_m"), "aoi.nominal_width_m")
    nominal_height = _require_number(aoi.get("nominal_height_m"), "aoi.nominal_height_m")
    tolerance = _require_number(aoi.get("dimension_tolerance_m", 100.0), "aoi.dimension_tolerance_m")
    dimension_ok = abs(width_m - nominal_width) <= tolerance and abs(height_m - nominal_height) <= tolerance
    buffer_m = _require_number(aoi.get("data_buffer_m", 0.0), "aoi.data_buffer_m")
    query_bbox = _buffer_bbox(bbox, buffer_m)

    sources = [_inspect_source(source, workspace_root, query_bbox) for source in config.get("sources", [])]
    configured_blockers = [str(value) for value in config.get("blocking_reasons", [])]
    missing_kind_blockers = [
        f"{source['id']}: missing {', '.join(source['missing_required_kinds'])}"
        for source in sources
        if source["missing_required_kinds"]
    ]
    coverage_blockers = [
        f"{source['id']}: AOI coverage is {source['aoi_coverage']}"
        for source in sources
        if source["aoi_coverage"] != "complete"
    ]
    blockers = configured_blockers + missing_kind_blockers + coverage_blockers
    if not dimension_ok:
        blockers.append("AOI polygon dimensions are outside the configured tolerance")

    mapray = config.get("mapray", {})
    max_files = int(mapray.get("max_files_per_dataset", 1000))
    bldg_assets = sum(
        record["asset_file_count"]
        for source in sources
        for record in source["records"]
        if record["kind"] == "bldg"
    )
    return {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "mode": "dry-run",
        "aoi": {
            "operations_geojson": operations_value,
            "feature_id": feature_id,
            "bbox": bbox,
            "buffered_source_query_bbox": query_bbox,
            "measured_width_m": round(width_m, 1),
            "measured_height_m": round(height_m, 1),
            "nominal_width_m": nominal_width,
            "nominal_height_m": nominal_height,
            "dimension_tolerance_m": tolerance,
            "dimension_check": "pass" if dimension_ok else "fail",
            "data_buffer_m": buffer_m,
        },
        "local_detail": config.get("local_detail", {}),
        "sources": sources,
        "mapray_upload_estimate": {
            "status": mapray.get("status", "planned"),
            "selected_building_asset_files": bldg_assets,
            "max_files_per_dataset": max_files,
            "minimum_dataset_count_before_citygml_split": (
                math.ceil(bldg_assets / max_files) if bldg_assets else 0
            ),
            "note": "Actual count may increase after CityGML texture-safe splitting.",
        },
        "ready_for_mapray_upload": len(blockers) == 0,
        "blocking_reasons": blockers,
        "next_action": (
            "prepare-and-validate-upload-bundles"
            if not blockers
            else "add-missing-municipality-sources-and-rerun-dry-run"
        ),
    }
