import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "city_pipeline"))

import gml_indexer
import gml_lod1_extract
from city_pipeline_api import (
    compare_building_positions,
    filter_citygml,
    generate_buildings_mjcf,
    integrate_mjcf,
    load_scenario,
)
from terrain_pipeline import (
    TerrainGrid,
    integrate_terrain_mjcf,
    validate_geotiff,
    validate_mujoco_hfield,
    write_hfield_binary,
    write_mapray_geotiff,
)


class CityPipelineTests(unittest.TestCase):
    def test_selected_building_can_use_measured_height_for_collision_proxy(self):
        gml = """<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"
 xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
 xmlns:gml="http://www.opengis.net/gml">
 <core:cityObjectMember><bldg:Building gml:id="tower">
  <bldg:measuredHeight uom="m">332.1</bldg:measuredHeight>
  <bldg:lod1Solid><gml:Solid><gml:exterior><gml:CompositeSurface>
   <gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>
    <gml:posList>35.0 139.0 18.4 35.0 139.001 18.4 35.001 139.0 18.4 35.0 139.0 18.4</gml:posList>
   </gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>
   <gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>
    <gml:posList>35.0 139.0 60.3 35.0 139.001 60.3 35.001 139.0 60.3 35.0 139.0 60.3</gml:posList>
   </gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>
  </gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod1Solid>
 </bldg:Building></core:cityObjectMember>
</core:CityModel>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tower_bldg_6697_op.gml"
            path.write_text(gml, encoding="utf-8")
            regular = gml_lod1_extract.extract_buildings_lod1(path)
            proxy = gml_lod1_extract.extract_buildings_lod1(
                path, measured_height_ids={"tower"}
            )
            self.assertAlmostEqual(60.3, regular[0]["zmax"])
            self.assertAlmostEqual(350.5, proxy[0]["zmax"])
            self.assertEqual("measuredHeight", proxy[0]["height_source"])
            self.assertAlmostEqual(60.3, proxy[0]["lod1_zmax"])

    def test_indexer_extracts_only_the_kind_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            gml = Path(directory) / "udx" / "bldg" / "53393567_bldg_6697_op.gml"
            gml.parent.mkdir(parents=True)
            gml.write_text("<root/>", encoding="utf-8")
            records = gml_indexer.build_index(Path(directory))
            self.assertEqual(records[0]["mesh"], "53393567")
            self.assertEqual(records[0]["kind"], "bldg")

    def test_load_scenario_rejects_missing_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps({"name": "x"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                load_scenario(path)

    def test_filter_refuses_unmanaged_existing_output(self):
        scenario = {
            "origin": {"latitude": 35.0, "longitude": 139.0},
            "extent": {"north_south_half_m": 10, "east_west_half_m": 10},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "citygml-extracted"
            (source / "udx").mkdir(parents=True)
            output.mkdir()
            (output / "user-file.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unmanaged output"):
                filter_citygml(source, output, scenario)
            self.assertTrue((output / "user-file.txt").exists())

    def test_integration_replaces_only_building_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.xml"
            buildings = root / "buildings.xml"
            clean = root / "clean.xml"
            integrated = root / "integrated.xml"
            base.write_text(
                '<mujoco><worldbody><body name="drone_base"/>'
                '<body name="body_bldg_old"><geom name="geom_bldg_old"/></body>'
                '</worldbody></mujoco>',
                encoding="utf-8",
            )
            buildings.write_text(
                '<mujoco><worldbody><body name="body_bldg_new">'
                '<geom name="geom_bldg_new" type="box" size="1 1 1"/>'
                '</body></worldbody></mujoco>',
                encoding="utf-8",
            )
            result = integrate_mjcf(base, buildings, clean, integrated)
            text = integrated.read_text(encoding="utf-8")
            self.assertEqual(result["removed_existing_building_nodes"], 1)
            self.assertEqual(result["added_building_nodes"], 1)
            self.assertIn('name="drone_base"', text)
            self.assertNotIn("body_bldg_old", text)
            self.assertIn("body_bldg_new", text)

    def test_building_generation_rebases_citygml_elevation(self):
        obb_data = {
            "results": [
                {
                    "id": "low",
                    "center": [10.0, 20.0],
                    "half_size": [2.0, 3.0],
                    "yaw": 0.0,
                    "zmin": 15.0,
                    "zmax": 25.0,
                },
                {
                    "id": "high",
                    "center": [30.0, 40.0],
                    "half_size": [4.0, 5.0],
                    "yaw": 0.0,
                    "zmin": 18.0,
                    "zmax": 30.0,
                },
            ]
        }
        scenario = {
            "name": "test",
            "vertical": {"z_offset": 2.0},
            "collision": {"mode": "drone"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "buildings.xml"
            generate_buildings_mjcf(obb_data, output, scenario)
            root = __import__("xml.etree.ElementTree", fromlist=["ElementTree"]).parse(output).getroot()
            geoms = {geom.get("name"): geom for geom in root.findall(".//geom")}
            low_pos = [float(value) for value in geoms["geom_low"].get("pos").split()]
            high_pos = [float(value) for value in geoms["geom_high"].get("pos").split()]
            self.assertAlmostEqual(low_pos[2], 7.0)
            self.assertAlmostEqual(high_pos[2], 11.0)

    def test_position_comparison_reports_vertical_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.xml"
            candidate = root / "candidate.xml"
            reference.write_text(
                '<mujoco><worldbody><geom name="geom_bldg_a" pos="1 2 3"/></worldbody></mujoco>',
                encoding="utf-8",
            )
            candidate.write_text(
                '<mujoco><worldbody><geom name="geom_bldg_a" pos="4 6 15"/></worldbody></mujoco>',
                encoding="utf-8",
            )
            comparison = compare_building_positions(reference, candidate)
            self.assertAlmostEqual(comparison["horizontal_rms_m"], 5.0)
            self.assertAlmostEqual(comparison["vertical_rms_m"], 12.0)
            self.assertAlmostEqual(comparison["spatial_rms_m"], 13.0)

    def test_hfield_binary_preserves_mujoco_xy_orientation(self):
        scenario = {
            "origin": {"latitude": 35.6625, "longitude": 139.70625, "altitude": 0.0},
            "horizontal_crs": "EPSG:6677",
            "extent": {"north_south_half_m": 1.0, "east_west_half_m": 1.0},
            "terrain": {"base_depth_m": 1.0},
        }
        values = np.asarray(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            dtype=np.float32,
        )
        grid = TerrainGrid(
            absolute_heights=values,
            model_heights=values,
            x_coordinates=np.linspace(-1.0, 1.0, 3),
            y_coordinates=np.linspace(-1.0, 1.0, 3),
            z_baseline_m=0.0,
            initially_missing_cells=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xml"
            hfield = root / "terrain.hfield"
            integrated = root / "integrated.xml"
            source.write_text(
                '<mujoco><asset><material name="white_ground"/></asset>'
                '<worldbody><geom name="ground" type="plane" size="10 10 .1"/></worldbody>'
                '</mujoco>',
                encoding="utf-8",
            )
            write_hfield_binary(hfield, values)
            integrate_terrain_mjcf(source, hfield, integrated, grid, scenario)
            validation = validate_mujoco_hfield(integrated, grid)
            self.assertEqual(validation["nhfield"], 1)
            self.assertLess(validation["max_absolute_error_m"], 1e-5)

    def test_mapray_geotiff_is_epsg4326_float32(self):
        scenario = {
            "origin": {"latitude": 35.6625, "longitude": 139.70625, "altitude": 0.0},
            "horizontal_crs": "EPSG:6677",
            "extent": {"north_south_half_m": 100.0, "east_west_half_m": 100.0},
            "terrain": {"nodata": -9999.0},
        }
        x = np.linspace(-100.0, 100.0, 17)
        y = np.linspace(-100.0, 100.0, 17)
        grid_x, grid_y = np.meshgrid(x, y)
        values = (20.0 + grid_x * 0.01 + grid_y * 0.02).astype(np.float32)
        grid = TerrainGrid(values, values, x, y, 0.0, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "terrain.tif"
            metadata = write_mapray_geotiff(output, grid, scenario)
            validation = validate_geotiff(output, grid, scenario)
            self.assertEqual(metadata["crs"], "EPSG:4326")
            self.assertEqual(metadata["dtype"], "float32")
            self.assertLess(validation["max_absolute_error_m"], 0.1)


if __name__ == "__main__":
    unittest.main()
