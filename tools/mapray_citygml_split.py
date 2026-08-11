#!/usr/bin/env python3
"""Split one textured CityGML into independently uploadable Mapray chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPOSITORY_ROOT / "src" / "city_pipeline"
sys.path.insert(0, str(MODULE_ROOT))

from citygml_texture_splitter import (  # noqa: E402
    analyze_citygml,
    build_split_chunks,
    split_citygml,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-textures", type=int, default=999)
    parser.add_argument(
        "--analyze-only", action="store_true", help="print the split plan without writing"
    )
    args = parser.parse_args()

    try:
        analysis = analyze_citygml(args.input)
        chunks = build_split_chunks(analysis, max_textures=args.max_textures)
        summary = {
            "source": str(analysis.source),
            "city_object_count": len(analysis.city_ids),
            "unique_texture_count": len(analysis.textures),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "index": chunk.index,
                    "city_object_count": len(chunk.city_objects),
                    "texture_count": len(chunk.texture_uris),
                }
                for chunk in chunks
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.analyze_only:
            manifest = split_citygml(
                analysis, args.output, max_textures=args.max_textures
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

