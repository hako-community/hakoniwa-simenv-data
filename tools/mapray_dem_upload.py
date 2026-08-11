#!/usr/bin/env python3
"""Upload a W6 GeoTIFF as a Mapray Cloud DEM Dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
MODULE_ROOT = REPOSITORY_ROOT / "src" / "city_pipeline"
sys.path.insert(0, str(MODULE_ROOT))

from mapray_b3d_uploader import MaprayUploadError, utc_now, write_state  # noqa: E402
from mapray_dem_uploader import MaprayDemApi, upload_geotiff  # noqa: E402


TOKEN_ENV = "MAPRAY_ORGANIZATION_TOKEN"
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"


def load_dotenv_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            return value.strip()
    return ""


def require_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip() or load_dotenv_value(
        DEFAULT_ENV_FILE, TOKEN_ENV
    )
    if not token:
        raise MaprayUploadError(
            f"{TOKEN_ENV} is not set in the process environment or {DEFAULT_ENV_FILE}"
        )
    return token


def upload(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    state_path = Path(args.state).resolve()
    if state_path.exists() and not args.replace_state:
        raise MaprayUploadError(
            f"state file already exists: {state_path}; use --replace-state to replace it"
        )
    api = MaprayDemApi(require_token(), timeout=args.timeout)
    dataset = api.create_dem_dataset(
        args.name,
        args.description,
        [{
            "display": args.attribution,
            "link": args.attribution_link,
            "description": "PLATEAU CityGML TINRelief converted in Phase W6",
        }],
    )
    dataset_id = str(dataset["id"])
    state: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_name": args.name,
        "status": "uploading",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input": {"name": input_path.name, "bytes": input_path.stat().st_size},
    }
    write_state(state_path, state)
    print(f"[CREATED] DEM Dataset ID: {dataset_id}")
    try:
        upload_geotiff(api, dataset_id, input_path, args.retries, args.timeout)
        state["status"] = "conversion_requested"
        state["updated_at"] = utc_now()
        write_state(state_path, state)
        api.convert_dem(dataset_id)
        if args.wait:
            deadline = time.monotonic() + args.wait_timeout
            while True:
                info = api.dem_dataset(dataset_id)
                dataset_status = str(info.get("status", "unknown"))
                error = str(info.get("error", ""))
                print(f"[STATUS] {dataset_status}" + (f": {error}" if error else ""))
                state["status"] = dataset_status
                state["updated_at"] = utc_now()
                if info.get("url"):
                    state["url"] = info["url"]
                if error:
                    state["error"] = error
                write_state(state_path, state)
                if dataset_status == "ready":
                    break
                if dataset_status in {"error", "failed"} or error:
                    raise MaprayUploadError(f"DEM conversion failed: {error or dataset_status}")
                if time.monotonic() >= deadline:
                    raise MaprayUploadError(
                        f"timed out waiting for conversion; Dataset ID remains {dataset_id}"
                    )
                time.sleep(args.poll_interval)
        print(f"[OK] DEM Dataset ID: {dataset_id}")
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["updated_at"] = utc_now()
        state["error"] = str(error)
        write_state(state_path, state)
        print(f"[FAILED DATASET] {dataset_id}", file=sys.stderr)
        raise


def status(args: argparse.Namespace) -> int:
    info = MaprayDemApi(require_token(), timeout=args.timeout).dem_dataset(args.dataset_id)
    safe = {
        key: info.get(key)
        for key in ("id", "name", "status", "error", "url", "fileinfo", "created_at", "updated_at")
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--input", required=True)
    upload_parser.add_argument("--name", default="shibuya-terrain-w6")
    upload_parser.add_argument(
        "--description", default="PLATEAU Shibuya TINRelief converted to EPSG:4326 Float32 GeoTIFF"
    )
    upload_parser.add_argument("--attribution", default="PLATEAU 3D都市モデル（渋谷区）")
    upload_parser.add_argument("--attribution-link", default="https://www.mlit.go.jp/plateau/")
    upload_parser.add_argument("--timeout", type=float, default=120.0)
    upload_parser.add_argument("--retries", type=int, default=3)
    upload_parser.add_argument("--wait", action="store_true")
    upload_parser.add_argument("--wait-timeout", type=float, default=3600.0)
    upload_parser.add_argument("--poll-interval", type=float, default=10.0)
    upload_parser.add_argument(
        "--state",
        default=str(WORKSPACE_ROOT / "runtime/windows/generated/shibuya/mapray-dem-upload-state.json"),
    )
    upload_parser.add_argument("--replace-state", action="store_true")
    upload_parser.set_defaults(handler=upload)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--dataset-id", required=True)
    status_parser.add_argument("--timeout", type=float, default=60.0)
    status_parser.set_defaults(handler=status)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
