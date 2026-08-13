from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "city_pipeline"))

from wide_area_planner import build_wide_area_plan


class WideAreaPlannerTests(unittest.TestCase):
    def test_plan_selects_intersecting_gml_and_blocks_partial_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            operations = workspace / "operations.geojson"
            operations.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"id": "aoi"},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[
                                        [139.694, 35.650],
                                        [139.719, 35.650],
                                        [139.719, 35.675],
                                        [139.694, 35.675],
                                        [139.694, 35.650],
                                    ]],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source = workspace / "plateau"
            bldg = source / "udx" / "bldg" / "53393586_bldg_test_op.gml"
            dem = source / "udx" / "dem" / "53393586_dem_test_op.gml"
            bldg.parent.mkdir(parents=True)
            dem.parent.mkdir(parents=True)
            bldg.write_text("<root/>", encoding="utf-8")
            dem.write_text("<root/>", encoding="utf-8")
            texture = bldg.parent / "53393586_bldg_test_appearance" / "texture.jpg"
            texture.parent.mkdir()
            texture.write_bytes(b"image")
            config = workspace / "wide.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenario_id": "test",
                        "aoi": {
                            "operations_geojson": "operations.geojson",
                            "feature_id": "aoi",
                            "nominal_width_m": 2250,
                            "nominal_height_m": 2783,
                            "dimension_tolerance_m": 100,
                            "data_buffer_m": 0,
                        },
                        "local_detail": {"width_m": 600, "height_m": 600},
                        "sources": [
                            {
                                "id": "source",
                                "root": "plateau",
                                "aoi_coverage": "partial",
                                "required_kinds": ["bldg", "dem"],
                            }
                        ],
                        "mapray": {"max_files_per_dataset": 1000},
                    }
                ),
                encoding="utf-8",
            )

            plan = build_wide_area_plan(config, workspace)

            self.assertEqual("pass", plan["aoi"]["dimension_check"])
            self.assertEqual({"bldg": 1, "dem": 1}, plan["sources"][0]["counts_by_kind"])
            self.assertEqual(3, plan["sources"][0]["selected_asset_file_count"])
            self.assertFalse(plan["ready_for_mapray_upload"])
            self.assertIn("AOI coverage is partial", " ".join(plan["blocking_reasons"]))


if __name__ == "__main__":
    unittest.main()
