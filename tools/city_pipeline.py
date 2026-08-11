#!/usr/bin/env python3
"""Windows-native command line entry point for the PLATEAU city pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
MODULE_ROOT = REPOSITORY_ROOT / "src" / "city_pipeline"
sys.path.insert(0, str(MODULE_ROOT))

from city_pipeline_api import (  # noqa: E402
    compare_building_positions,
    convert_lod1_to_obb,
    extract_lod1,
    filter_citygml,
    generate_buildings_mjcf,
    integrate_mjcf,
    load_scenario,
    sha256_file,
    validate_mjcf,
)
from terrain_pipeline import (  # noqa: E402
    compare_building_bases,
    extract_tin,
    integrate_terrain_mjcf,
    interpolate_terrain_grid,
    select_dem_sources,
    validate_geotiff,
    validate_mujoco_hfield,
    write_browser_terrain_grid,
    write_hfield_binary,
    write_mapray_geotiff,
)


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def convert(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    source_root = resolve_workspace_path(args.input)
    scenario_path = resolve_workspace_path(args.scenario)
    output_root = resolve_workspace_path(args.output)
    scenario = load_scenario(scenario_path)
    output_root.mkdir(parents=True, exist_ok=True)

    extracted_root = output_root / "citygml-extracted"
    lod1_path = output_root / "buildings-lod1.json"
    obb_path = output_root / "buildings-obb.json"
    buildings_path = output_root / "buildings.xml"
    base_output = output_root / "drone-base.xml"
    integrated_output = output_root / "drone.xml"
    manifest_path = output_root / "manifest.json"

    print(f"[1/6] Filter CityGML: {source_root}")
    selection = filter_citygml(source_root, extracted_root, scenario)
    print(f"      selected {len(selection['selected'])} building GML file(s)")

    print("[2/6] Extract LOD1 footprints")
    lod1 = extract_lod1(extracted_root, lod1_path, scenario)
    print(f"      extracted {len(lod1['polygons'])} building footprint(s)")

    print("[3/6] Convert footprints to collision boxes")
    obb = convert_lod1_to_obb(lod1, obb_path, scenario)
    print(f"      generated {len(obb['results'])} collision record(s)")

    print("[4/6] Generate buildings.xml")
    building_counts = generate_buildings_mjcf(obb, buildings_path, scenario, args.collide)

    base_model_value = args.base_model or scenario.get("base_model")
    if not base_model_value:
        raise ValueError("base model must be supplied by --base-model or scenario.base_model")
    base_model = resolve_workspace_path(base_model_value)
    print(f"[5/6] Integrate with base model: {base_model}")
    integration = integrate_mjcf(base_model, buildings_path, base_output, integrated_output)

    validation = None
    if not args.no_validate:
        print("[6/6] Compile integrated MJCF with Python MuJoCo")
        validation = validate_mjcf(integrated_output)
        print(
            f"      PASS nbody={validation['nbody']} ngeom={validation['ngeom']} "
            f"buildings={validation['building_geom_count']}"
        )
    else:
        print("[6/6] MuJoCo validation skipped")

    comparison = None
    reference_value = args.reference_model or scenario.get("reference_model")
    if reference_value:
        reference_model = resolve_workspace_path(reference_value)
        comparison = compare_building_positions(reference_model, integrated_output)
        if comparison.get("matched_geom_count"):
            print(
                "      reference position comparison: "
                f"matched={comparison['matched_geom_count']} "
                f"horizontal_rms={comparison['horizontal_rms_m']:.3f}m "
                f"vertical_rms={comparison['vertical_rms_m']:.3f}m "
                f"spatial_rms={comparison['spatial_rms_m']:.3f}m"
            )

    generated_files = [lod1_path, obb_path, buildings_path, base_output, integrated_output]
    manifest = {
        "schema_version": 1,
        "scenario": scenario,
        "source_root": str(source_root),
        "selection": selection,
        "building_counts": building_counts,
        "integration": integration,
        "validation": validation,
        "reference_comparison": comparison,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in generated_files
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] W4 output: {output_root}")
    print(f"[OK] manifest: {manifest_path}")
    return 0


def terrain(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    source_root = resolve_workspace_path(args.input)
    scenario_path = resolve_workspace_path(args.scenario)
    output_root = resolve_workspace_path(args.output)
    scenario = load_scenario(scenario_path)
    output_root.mkdir(parents=True, exist_ok=True)

    lod1_path = resolve_workspace_path(args.buildings_lod1) if args.buildings_lod1 else output_root / "buildings-lod1.json"
    model_path = resolve_workspace_path(args.model) if args.model else output_root / "drone.xml"
    hfield_path = output_root / "terrain.hfield"
    geotiff_path = output_root / "terrain-mapray.tif"
    browser_grid_path = output_root / "terrain-grid.json"
    integrated_path = output_root / "drone-terrain.xml"
    manifest_path = output_root / "terrain-manifest.json"
    for required in (lod1_path, model_path):
        if not required.is_file():
            raise ValueError(f"W6 prerequisite is missing: {required}")

    print(f"[1/7] Select PLATEAU DEM CityGML: {source_root}")
    sources = select_dem_sources(source_root, scenario)
    print(f"      selected {len(sources)} DEM file(s)")

    print("[2/7] Extract scenario TIN triangles")
    tin = extract_tin(sources, scenario)
    print(
        f"      selected {tin.selected_triangle_count} of "
        f"{tin.scanned_triangle_count} triangle(s)"
    )

    buildings = json.loads(lod1_path.read_text(encoding="utf-8"))["polygons"]
    zmin_values = [float(item["zmin"]) for item in buildings if "zmin" in item]
    if not zmin_values:
        raise ValueError("buildings-lod1.json contains no zmin values")
    baseline_mode = scenario.get("terrain", {}).get("z_baseline", "building_min")
    if baseline_mode != "building_min":
        raise ValueError("W6 currently supports terrain.z_baseline='building_min'")
    z_baseline = min(zmin_values)

    print(f"[3/7] Interpolate terrain grid (z baseline={z_baseline:.4f}m)")
    grid = interpolate_terrain_grid(tin, scenario, z_baseline)
    print(
        f"      grid={grid.columns}x{grid.rows} "
        f"absolute={grid.absolute_heights.min():.3f}..{grid.absolute_heights.max():.3f}m"
    )

    print("[4/7] Write Mapray GeoTIFF and MuJoCo HField")
    geotiff = write_mapray_geotiff(geotiff_path, grid, scenario)
    write_hfield_binary(hfield_path, grid.model_heights)
    browser_grid = write_browser_terrain_grid(browser_grid_path, grid)

    print(f"[5/7] Replace plane ground in MJCF: {model_path}")
    integration = integrate_terrain_mjcf(
        model_path, hfield_path, integrated_path, grid, scenario
    )

    print("[6/7] Validate Mapray GeoTIFF and building base heights")
    geotiff_validation = validate_geotiff(geotiff_path, grid, scenario)
    building_comparison = compare_building_bases(lod1_path, grid)
    print(
        f"      GeoTIFF RMS={geotiff_validation['rms_error_m']:.3f}m, "
        f"buildings within 2m={building_comparison.get('within_2m_ratio', 0):.1%}"
    )

    mujoco_validation = None
    if not args.no_validate:
        print("[7/7] Compile and ray-test MuJoCo HField")
        mujoco_validation = validate_mujoco_hfield(integrated_path, grid)
        tolerance = float(scenario.get("terrain", {}).get("validation_tolerance_m", 0.05))
        if mujoco_validation["max_absolute_error_m"] > tolerance:
            raise ValueError(
                "MuJoCo HField orientation/height validation failed: "
                f"max error {mujoco_validation['max_absolute_error_m']:.6f}m > {tolerance:.6f}m"
            )
        print(
            f"      PASS MuJoCo {mujoco_validation['mujoco_version']} "
            f"max error={mujoco_validation['max_absolute_error_m']:.6f}m"
        )
    else:
        print("[7/7] MuJoCo validation skipped")

    generated_files = [geotiff_path, hfield_path, browser_grid_path, integrated_path]
    manifest = {
        "schema_version": 1,
        "scenario": scenario,
        "source_root": str(source_root),
        "source_files": [str(path) for path in sources],
        "tin": {
            "scanned_triangle_count": tin.scanned_triangle_count,
            "selected_triangle_count": tin.selected_triangle_count,
            "vertex_count": int(tin.vertices.shape[0]),
        },
        "grid": {
            "rows": grid.rows,
            "columns": grid.columns,
            "initially_missing_cells": grid.initially_missing_cells,
            "z_baseline_m": grid.z_baseline_m,
            "minimum_absolute_height_m": float(grid.absolute_heights.min()),
            "maximum_absolute_height_m": float(grid.absolute_heights.max()),
            "minimum_model_height_m": grid.minimum_model_height,
            "maximum_model_height_m": grid.maximum_model_height,
        },
        "mapray_geotiff": geotiff,
        "browser_terrain_grid": browser_grid,
        "integration": integration,
        "geotiff_validation": geotiff_validation,
        "building_terrain_comparison": building_comparison,
        "mujoco_validation": mujoco_validation,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in generated_files
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] W6 output: {output_root}")
    print(f"[OK] manifest: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert", help="convert CityGML and integrate MJCF")
    convert_parser.add_argument("--input", required=True, help="extracted PLATEAU CityGML root")
    convert_parser.add_argument("--scenario", required=True, help="pipeline scenario JSON")
    convert_parser.add_argument("--output", required=True, help="generated output directory")
    convert_parser.add_argument("--base-model", help="base drone MJCF; overrides scenario")
    convert_parser.add_argument("--reference-model", help="reference MJCF used for origin comparison")
    convert_parser.add_argument("--collide", choices=("all", "drone", "none"), help="collision mode override")
    convert_parser.add_argument("--no-validate", action="store_true", help="skip Python MuJoCo compilation")
    convert_parser.set_defaults(handler=convert)
    terrain_parser = subparsers.add_parser(
        "terrain", help="convert PLATEAU TIN relief to Mapray GeoTIFF and MuJoCo HField"
    )
    terrain_parser.add_argument("--input", required=True, help="PLATEAU CityGML root")
    terrain_parser.add_argument("--scenario", required=True, help="pipeline scenario JSON")
    terrain_parser.add_argument("--output", required=True, help="generated output directory")
    terrain_parser.add_argument("--model", help="W4 integrated MJCF; defaults to output/drone.xml")
    terrain_parser.add_argument(
        "--buildings-lod1", help="W4 LOD1 JSON; defaults to output/buildings-lod1.json"
    )
    terrain_parser.add_argument("--no-validate", action="store_true", help="skip MuJoCo ray validation")
    terrain_parser.set_defaults(handler=terrain)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
