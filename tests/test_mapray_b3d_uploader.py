from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "city_pipeline"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from mapray_b3d_uploader import MaprayUploadError, build_upload_plan, write_state
from mapray_dem_uploader import MaprayDemApi
from mapray_b3d_upload import load_dotenv_value


class MaprayB3dUploaderTests(unittest.TestCase):
    def make_gml(self, root: Path, image_names: list[str]) -> None:
        body = "".join(
            f"<app:imageURI>{name}</app:imageURI>" for name in image_names
        )
        (root / "sample.gml").write_text(
            '<core:CityModel xmlns:core="urn:core" xmlns:app="urn:app">'
            + body
            + "</core:CityModel>",
            encoding="utf-8",
        )

    def test_plan_preserves_relative_texture_paths_and_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f"appearance/texture-{index}.jpg" for index in range(5)]
            (root / "appearance").mkdir()
            for name in names:
                (root / name).write_bytes(b"jpeg")
            self.make_gml(root, names)

            plan = build_upload_plan(root, batch_size=3)

            self.assertEqual(plan.gml_count, 1)
            self.assertEqual(plan.texture_count, 5)
            self.assertEqual([len(batch) for batch in plan.batches], [3, 3])
            self.assertEqual(plan.files[0].name, "sample.gml")
            self.assertEqual(plan.files[1].name, "appearance/texture-0.jpg")

    def test_plan_rejects_missing_referenced_texture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_gml(root, ["appearance/missing.jpg"])
            with self.assertRaisesRegex(MaprayUploadError, "missing texture"):
                build_upload_plan(root)

    def test_dotenv_loader_supports_quotes_without_exposing_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# local secret\nMAPRAY_ORGANIZATION_TOKEN='secret-value'\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_dotenv_value(path, "MAPRAY_ORGANIZATION_TOKEN"),
                "secret-value",
            )

    def test_state_file_contains_only_supplied_safe_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = {"dataset_id": "123", "status": "uploading"}
            write_state(path, state)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), state)

    def test_project_ignores_dotenv(self) -> None:
        gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore.splitlines())

    def test_dem_dataset_request_uses_geotiff_contract(self) -> None:
        calls = []

        class FakeDemApi(MaprayDemApi):
            def __init__(self):
                pass

            def _json_request(self, method, path, payload=None):
                calls.append((method, path, payload))
                return {"id": "dem-1"}

        dataset = FakeDemApi().create_dem_dataset("terrain", "description", [])
        self.assertEqual(dataset["id"], "dem-1")
        self.assertEqual(calls[0][0:2], ("POST", "/demdatasets/v2"))
        self.assertEqual(calls[0][2]["src_file_type"], "geotiff")
        self.assertEqual(calls[0][2]["dst_file_type"], "dem")


if __name__ == "__main__":
    unittest.main()
