from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "city_pipeline"))

from citygml_texture_splitter import analyze_citygml, build_split_chunks, split_citygml
from mapray_b3d_uploader import build_upload_plan


class CityGmlTextureSplitterTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        appearance = root / "appearance"
        appearance.mkdir()
        for name in ("shared.jpg", "a.jpg", "b.jpg"):
            (appearance / name).write_bytes(b"jpeg")
        source = root / "sample.gml"
        source.write_text(
            '''<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"
 xmlns:gml="http://www.opengis.net/gml"
 xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
 xmlns:app="http://www.opengis.net/citygml/appearance/2.0">
 <gml:boundedBy><gml:Envelope><gml:lowerCorner>0 0 0</gml:lowerCorner><gml:upperCorner>1 1 1</gml:upperCorner></gml:Envelope></gml:boundedBy>
 <core:cityObjectMember><bldg:Building gml:id="b1"><gml:Polygon gml:id="p1"/></bldg:Building></core:cityObjectMember>
 <core:cityObjectMember><bldg:Building gml:id="b2"><gml:Polygon gml:id="p2"/></bldg:Building></core:cityObjectMember>
 <app:appearanceMember><app:Appearance><app:theme>rgbTexture</app:theme>
  <app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>appearance/shared.jpg</app:imageURI><app:target uri="#p1"/><app:target uri="#p2"/></app:ParameterizedTexture></app:surfaceDataMember>
  <app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>appearance/a.jpg</app:imageURI><app:target uri="#p1"/></app:ParameterizedTexture></app:surfaceDataMember>
  <app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>appearance/b.jpg</app:imageURI><app:target uri="#p2"/></app:ParameterizedTexture></app:surfaceDataMember>
 </app:Appearance></app:appearanceMember>
</core:CityModel>
''',
            encoding="utf-8",
        )
        return source

    def test_analyze_and_split_with_shared_texture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            analysis = analyze_citygml(source)
            chunks = build_split_chunks(analysis, max_textures=2)

            self.assertEqual(len(analysis.city_ids), 2)
            self.assertEqual(len(analysis.textures), 3)
            self.assertEqual([len(chunk.texture_uris) for chunk in chunks], [2, 2])

            output = root / "chunks"
            split_citygml(analysis, output, max_textures=2)
            for chunk_dir in sorted(output.glob("chunk-*")):
                plan = build_upload_plan(chunk_dir)
                self.assertEqual(plan.gml_count, 1)
                self.assertEqual(plan.texture_count, 2)
                self.assertEqual(len(plan.files), 3)
                ET.parse(next(chunk_dir.glob("*.gml")))


if __name__ == "__main__":
    unittest.main()

