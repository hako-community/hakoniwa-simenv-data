"""PLATEAU TIN terrain conversion for Mapray GeoTIFF and MuJoCo HField."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import matplotlib.tri as mtri
import mujoco
import numpy as np
import rasterio
from lxml import etree
from pyproj import Transformer
from rasterio.fill import fillnodata
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

import gml_extract
import gml_indexer


GML_POS_LIST = "{http://www.opengis.net/gml}posList"
TERRAIN_ASSET_NAME = "terrain"
TERRAIN_GEOM_NAME = "ground"
TERRAIN_GEOM_GROUP = 3


@dataclass
class TinData:
    vertices: np.ndarray
    triangles: np.ndarray
    source_files: tuple[Path, ...]
    scanned_triangle_count: int
    selected_triangle_count: int


@dataclass
class TerrainGrid:
    absolute_heights: np.ndarray
    model_heights: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    z_baseline_m: float
    initially_missing_cells: int

    @property
    def rows(self) -> int:
        return int(self.absolute_heights.shape[0])

    @property
    def columns(self) -> int:
        return int(self.absolute_heights.shape[1])

    @property
    def minimum_model_height(self) -> float:
        return float(np.min(self.model_heights))

    @property
    def maximum_model_height(self) -> float:
        return float(np.max(self.model_heights))


def select_dem_sources(source_root: Path, scenario: dict[str, Any]) -> tuple[Path, ...]:
    origin = scenario["origin"]
    extent = scenario["extent"]
    margin = float(scenario.get("terrain", {}).get("source_margin_m", 20.0))
    query = gml_extract.make_query_bbox(
        float(origin["latitude"]),
        float(origin["longitude"]),
        float(extent["north_south_half_m"]) + margin,
        float(extent["east_west_half_m"]) + margin,
    )
    selected: list[Path] = []
    for record in gml_indexer.build_index(source_root):
        if record["kind"] != "dem":
            continue
        if gml_extract.is_intersect(query, gml_extract.mesh_to_bbox(record["mesh"])):
            selected.append(Path(record["path"]))
    if not selected:
        raise ValueError(f"no PLATEAU DEM CityGML intersects the scenario: {source_root}")
    return tuple(sorted(selected))


def _iter_triangle_vertices(path: Path) -> Iterable[tuple[tuple[float, float, float], ...]]:
    context = etree.iterparse(
        str(path), events=("end",), tag=GML_POS_LIST, huge_tree=True
    )
    for _, element in context:
        text = element.text or ""
        values = np.fromstring(text, dtype=np.float64, sep=" ")
        if values.size >= 9 and values.size % 3 == 0:
            points = values.reshape((-1, 3))
            yield tuple(tuple(float(value) for value in point) for point in points[:3])
        element.clear()
        parent = element.getparent()
        if parent is not None:
            while element.getprevious() is not None:
                del parent[0]
    del context


def extract_tin(
    source_files: Iterable[Path],
    scenario: dict[str, Any],
) -> TinData:
    origin = scenario["origin"]
    extent = scenario["extent"]
    margin = float(scenario.get("terrain", {}).get("source_margin_m", 20.0))
    query = gml_extract.make_query_bbox(
        float(origin["latitude"]),
        float(origin["longitude"]),
        float(extent["north_south_half_m"]) + margin,
        float(extent["east_west_half_m"]) + margin,
    )

    raw_vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    vertex_ids: dict[tuple[float, float, float], int] = {}
    scanned = 0
    selected = 0
    files = tuple(source_files)
    for source in files:
        for points in _iter_triangle_vertices(source):
            scanned += 1
            latitudes = [point[0] for point in points]
            longitudes = [point[1] for point in points]
            if (
                max(latitudes) < query["min_lat"]
                or min(latitudes) > query["max_lat"]
                or max(longitudes) < query["min_lon"]
                or min(longitudes) > query["max_lon"]
            ):
                continue
            indices: list[int] = []
            for point in points:
                key = (round(point[0], 12), round(point[1], 12), round(point[2], 6))
                index = vertex_ids.get(key)
                if index is None:
                    index = len(raw_vertices)
                    vertex_ids[key] = index
                    raw_vertices.append(point)
                indices.append(index)
            if len(set(indices)) == 3:
                triangles.append(tuple(indices))
                selected += 1

    if not triangles:
        raise ValueError("no DEM triangles intersect the requested terrain extent")

    raw = np.asarray(raw_vertices, dtype=np.float64)
    transformer = Transformer.from_crs(4326, scenario["horizontal_crs"], always_xy=True)
    east, north = transformer.transform(raw[:, 1], raw[:, 0])
    origin_east, origin_north = transformer.transform(
        float(origin["longitude"]), float(origin["latitude"])
    )
    # MuJoCo axes: x=north, y=-east, z=up.
    projected = np.column_stack((north - origin_north, -(east - origin_east), raw[:, 2]))
    triangle_array = np.asarray(triangles, dtype=np.int32)
    a = projected[triangle_array[:, 0], :2]
    b = projected[triangle_array[:, 1], :2]
    c = projected[triangle_array[:, 2], :2]
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    triangle_array = triangle_array[area2 > 1e-8]
    if triangle_array.size == 0:
        raise ValueError("all selected DEM triangles were degenerate after projection")
    return TinData(projected, triangle_array, files, scanned, int(len(triangle_array)))


def interpolate_terrain_grid(
    tin: TinData,
    scenario: dict[str, Any],
    z_baseline_m: float,
) -> TerrainGrid:
    terrain_config = scenario.get("terrain", {})
    grid_size = int(terrain_config.get("grid_size", 257))
    if grid_size < 3:
        raise ValueError("terrain.grid_size must be at least 3")
    north_half = float(scenario["extent"]["north_south_half_m"])
    east_half = float(scenario["extent"]["east_west_half_m"])
    x_coordinates = np.linspace(-north_half, north_half, grid_size)
    y_coordinates = np.linspace(-east_half, east_half, grid_size)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)

    triangulation = mtri.Triangulation(
        tin.vertices[:, 0], tin.vertices[:, 1], tin.triangles
    )
    interpolator = mtri.LinearTriInterpolator(triangulation, tin.vertices[:, 2])
    interpolated = interpolator(grid_x, grid_y)
    mask = np.ma.getmaskarray(interpolated) | ~np.isfinite(np.ma.filled(interpolated, np.nan))
    missing = int(np.count_nonzero(mask))
    values = np.ma.filled(interpolated, 0.0).astype(np.float32)
    if missing:
        values = fillnodata(
            values,
            mask=(~mask).astype(np.uint8),
            max_search_distance=float(grid_size),
            smoothing_iterations=0,
        ).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("DEM interpolation left non-finite height cells")
    return TerrainGrid(
        absolute_heights=values,
        model_heights=(values - np.float32(z_baseline_m)).astype(np.float32),
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        z_baseline_m=float(z_baseline_m),
        initially_missing_cells=missing,
    )


def write_hfield_binary(path: Path, model_heights: np.ndarray) -> None:
    values = np.asarray(model_heights, dtype="<f4")
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("MuJoCo HField must be a two-dimensional grid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", values.shape[0], values.shape[1]))
        stream.write(values.tobytes(order="C"))


def write_mapray_geotiff(
    path: Path,
    grid: TerrainGrid,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    origin = scenario["origin"]
    transformer = Transformer.from_crs(4326, scenario["horizontal_crs"], always_xy=True)
    origin_east, origin_north = transformer.transform(
        float(origin["longitude"]), float(origin["latitude"])
    )
    north_half = float(scenario["extent"]["north_south_half_m"])
    east_half = float(scenario["extent"]["east_west_half_m"])
    x_spacing = 2.0 * north_half / (grid.columns - 1)
    y_spacing = 2.0 * east_half / (grid.rows - 1)
    source_bounds = (
        origin_east - east_half - y_spacing / 2.0,
        origin_north - north_half - x_spacing / 2.0,
        origin_east + east_half + y_spacing / 2.0,
        origin_north + north_half + x_spacing / 2.0,
    )
    source_transform = from_bounds(
        *source_bounds, width=grid.columns, height=grid.rows
    )
    # Grid axes are [model y, model x]. Convert to north-up [north, east].
    source_values = grid.absolute_heights.T[::-1, ::-1]
    destination_bounds = transform_bounds(
        scenario["horizontal_crs"], "EPSG:4326", *source_bounds, densify_pts=21
    )
    destination_transform = from_bounds(
        *destination_bounds, width=grid.columns, height=grid.rows
    )
    nodata = float(scenario.get("terrain", {}).get("nodata", -9999.0))
    destination = np.full(source_values.shape, nodata, dtype=np.float32)
    reproject(
        source=source_values,
        destination=destination,
        src_transform=source_transform,
        src_crs=scenario["horizontal_crs"],
        dst_transform=destination_transform,
        dst_crs="EPSG:4326",
        src_nodata=None,
        dst_nodata=nodata,
        resampling=Resampling.bilinear,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.columns,
        height=grid.rows,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=destination_transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
    ) as dataset:
        dataset.write(destination, 1)
        dataset.update_tags(
            AREA_OR_POINT="Area",
            VERTICAL_DATUM="JGD2011 (vertical) height / orthometric height",
            SOURCE="PLATEAU CityGML TINRelief EPSG:6697",
        )
    valid = destination[destination != nodata]
    return {
        "crs": "EPSG:4326",
        "dtype": "float32",
        "raster_type": "Area",
        "nodata": nodata,
        "width": grid.columns,
        "height": grid.rows,
        "bounds": [float(value) for value in destination_bounds],
        "valid_cell_count": int(valid.size),
        "minimum_height_m": float(np.min(valid)),
        "maximum_height_m": float(np.max(valid)),
    }


def write_browser_terrain_grid(path: Path, grid: TerrainGrid) -> dict[str, Any]:
    payload = {
        "schemaVersion": 1,
        "frame": "mujoco_x_north_y_minus_east_z_up",
        "rows": grid.rows,
        "columns": grid.columns,
        "xMinM": float(grid.x_coordinates[0]),
        "xMaxM": float(grid.x_coordinates[-1]),
        "yMinM": float(grid.y_coordinates[0]),
        "yMaxM": float(grid.y_coordinates[-1]),
        "zBaselineM": grid.z_baseline_m,
        "modelHeightsM": np.round(grid.model_heights, 4).ravel(order="C").tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": grid.rows,
        "columns": grid.columns,
        "height_count": grid.rows * grid.columns,
        "rounding_precision_m": 0.0001,
    }


def integrate_terrain_mjcf(
    model_path: Path,
    hfield_path: Path,
    output_path: Path,
    grid: TerrainGrid,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(model_path, parser=parser).getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"MJCF must contain asset and worldbody: {model_path}")
    for child in list(asset):
        if child.tag == "hfield" and child.get("name") == TERRAIN_ASSET_NAME:
            asset.remove(child)
    for child in list(worldbody):
        if child.tag == "geom" and child.get("name") == TERRAIN_GEOM_NAME:
            worldbody.remove(child)

    elevation_range = grid.maximum_model_height - grid.minimum_model_height
    if elevation_range <= 0:
        raise ValueError("terrain elevation range must be positive")
    terrain_config = scenario.get("terrain", {})
    base_depth = float(terrain_config.get("base_depth_m", 5.0))
    north_half = float(scenario["extent"]["north_south_half_m"])
    east_half = float(scenario["extent"]["east_west_half_m"])
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": TERRAIN_ASSET_NAME,
            "file": hfield_path.name,
            "size": f"{north_half:.9g} {east_half:.9g} {elevation_range:.9g} {base_depth:.9g}",
        },
    )
    terrain_geom = ET.Element(
        "geom",
        {
            "name": TERRAIN_GEOM_NAME,
            "type": "hfield",
            "hfield": TERRAIN_ASSET_NAME,
            "pos": f"0 0 {grid.minimum_model_height:.9g}",
            "material": "white_ground",
            "contype": "1",
            "conaffinity": "2",
            "group": str(TERRAIN_GEOM_GROUP),
            "friction": "0.8 0.1 0.1",
        },
    )
    first_body = next(
        (index for index, child in enumerate(list(worldbody)) if child.tag == "body"),
        len(worldbody),
    )
    worldbody.insert(first_body, terrain_geom)
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return {
        "removed_plane_ground": True,
        "hfield_asset": TERRAIN_ASSET_NAME,
        "hfield_geom": TERRAIN_GEOM_NAME,
        "hfield_rows": grid.rows,
        "hfield_columns": grid.columns,
        "minimum_model_height_m": grid.minimum_model_height,
        "maximum_model_height_m": grid.maximum_model_height,
        "elevation_range_m": elevation_range,
        "base_depth_m": base_depth,
    }


def _sample_grid(grid: TerrainGrid, x: float, y: float, absolute: bool = True) -> float:
    values = grid.absolute_heights if absolute else grid.model_heights
    x_index = (x - grid.x_coordinates[0]) / (grid.x_coordinates[-1] - grid.x_coordinates[0]) * (grid.columns - 1)
    y_index = (y - grid.y_coordinates[0]) / (grid.y_coordinates[-1] - grid.y_coordinates[0]) * (grid.rows - 1)
    if x_index < 0 or y_index < 0 or x_index > grid.columns - 1 or y_index > grid.rows - 1:
        raise ValueError("sample point is outside the terrain grid")
    x0 = min(int(math.floor(x_index)), grid.columns - 2)
    y0 = min(int(math.floor(y_index)), grid.rows - 2)
    dx = x_index - x0
    dy = y_index - y0
    return float(
        values[y0, x0] * (1 - dx) * (1 - dy)
        + values[y0, x0 + 1] * dx * (1 - dy)
        + values[y0 + 1, x0] * (1 - dx) * dy
        + values[y0 + 1, x0 + 1] * dx * dy
    )


def compare_building_bases(lod1_path: Path, grid: TerrainGrid) -> dict[str, Any]:
    polygons = json.loads(lod1_path.read_text(encoding="utf-8"))["polygons"]
    differences: list[float] = []
    for polygon in polygons:
        vertices = polygon.get("vertices") or []
        if not vertices or "zmin" not in polygon:
            continue
        east = sum(float(point[0]) for point in vertices) / len(vertices)
        north = sum(float(point[1]) for point in vertices) / len(vertices)
        try:
            terrain_height = _sample_grid(grid, north, -east, absolute=True)
        except ValueError:
            continue
        differences.append(float(polygon["zmin"]) - terrain_height)
    if not differences:
        return {"matched_building_count": 0}
    values = np.asarray(differences, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "matched_building_count": int(values.size),
        "building_base_minus_terrain_mean_m": float(np.mean(values)),
        "building_base_minus_terrain_median_m": float(np.median(values)),
        "absolute_rms_m": float(np.sqrt(np.mean(values * values))),
        "absolute_p95_m": float(np.percentile(absolute, 95)),
        "absolute_max_m": float(np.max(absolute)),
        "within_1m_ratio": float(np.mean(absolute <= 1.0)),
        "within_2m_ratio": float(np.mean(absolute <= 2.0)),
    }


def validate_geotiff(path: Path, grid: TerrainGrid, scenario: dict[str, Any]) -> dict[str, Any]:
    origin = scenario["origin"]
    to_geographic = Transformer.from_crs(scenario["horizontal_crs"], 4326, always_xy=True)
    to_projected = Transformer.from_crs(4326, scenario["horizontal_crs"], always_xy=True)
    origin_east, origin_north = to_projected.transform(
        float(origin["longitude"]), float(origin["latitude"])
    )
    samples = []
    offsets = (-0.75, 0.0, 0.75)
    north_half = float(scenario["extent"]["north_south_half_m"])
    east_half = float(scenario["extent"]["east_west_half_m"])
    points: list[tuple[float, float, float]] = []
    for north_ratio in offsets:
        for east_ratio in offsets:
            north = north_ratio * north_half
            east = east_ratio * east_half
            lon, lat = to_geographic.transform(origin_east + east, origin_north + north)
            points.append((lon, lat, _sample_grid(grid, north, -east, absolute=True)))
    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.crs.to_epsg() != 4326:
            raise ValueError("Mapray GeoTIFF must use EPSG:4326")
        if dataset.dtypes != ("float32",):
            raise ValueError("Mapray GeoTIFF must use Float32")
        for (lon, lat, expected), value in zip(points, dataset.sample([(p[0], p[1]) for p in points])):
            actual = float(value[0])
            samples.append({
                "longitude": lon,
                "latitude": lat,
                "expected_height_m": expected,
                "geotiff_height_m": actual,
                "error_m": actual - expected,
            })
    errors = np.asarray([item["error_m"] for item in samples], dtype=np.float64)
    return {
        "sample_count": len(samples),
        "rms_error_m": float(np.sqrt(np.mean(errors * errors))),
        "max_absolute_error_m": float(np.max(np.abs(errors))),
        "samples": samples,
    }


def validate_mujoco_hfield(model_path: Path, grid: TerrainGrid) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.zeros(6, dtype=np.uint8)
    group[TERRAIN_GEOM_GROUP] = 1
    geom_id = np.array([-1], dtype=np.int32)
    samples = []
    indices = (0, grid.rows // 2, grid.rows - 1)
    ray_start_z = grid.maximum_model_height + 100.0
    for row in indices:
        for column in indices:
            x = float(grid.x_coordinates[column])
            y = float(grid.y_coordinates[row])
            distance = mujoco.mj_ray(
                model,
                data,
                np.asarray([x, y, ray_start_z], dtype=np.float64),
                np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
                group,
                1,
                -1,
                geom_id,
            )
            if distance < 0:
                raise ValueError(f"MuJoCo ray missed HField at x={x}, y={y}")
            actual = ray_start_z - float(distance)
            expected = float(grid.model_heights[row, column])
            samples.append({
                "x_m": x,
                "y_m": y,
                "expected_height_m": expected,
                "mujoco_height_m": actual,
                "error_m": actual - expected,
            })
    errors = np.asarray([item["error_m"] for item in samples], dtype=np.float64)
    return {
        "mujoco_version": mujoco.__version__,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nhfield": int(model.nhfield),
        "hfield_rows": int(model.hfield_nrow[0]),
        "hfield_columns": int(model.hfield_ncol[0]),
        "sample_count": len(samples),
        "rms_error_m": float(np.sqrt(np.mean(errors * errors))),
        "max_absolute_error_m": float(np.max(np.abs(errors))),
        "samples": samples,
    }
