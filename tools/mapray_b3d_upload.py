#!/usr/bin/env python3
"""Plan or execute a batched Mapray Cloud Building Dataset upload."""

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

from mapray_b3d_uploader import (  # noqa: E402
    MaprayB3dApi,
    MaprayUploadError,
    build_upload_plan,
    upload_batch,
    utc_now,
    write_state,
)


TOKEN_ENV = "MAPRAY_ORGANIZATION_TOKEN"
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"
DEFAULT_ATTRIBUTION = "PLATEAU 3D都市モデル（渋谷区）"
DEFAULT_ATTRIBUTION_LINK = "https://www.mlit.go.jp/plateau/"


def resolve_cli_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve()


def print_plan(plan) -> None:
    summary = plan.as_dict()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plan_command(args: argparse.Namespace) -> int:
    plan = build_upload_plan(resolve_cli_path(args.input), args.batch_size)
    print_plan(plan)
    return 0


def load_dotenv_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            return value.strip()
    return ""


def require_token(env_file: Path = DEFAULT_ENV_FILE) -> str:
    token = os.environ.get(TOKEN_ENV, "").strip() or load_dotenv_value(env_file, TOKEN_ENV)
    if not token:
        raise MaprayUploadError(
            f"{TOKEN_ENV} is not set in the process environment or {env_file}"
        )
    return token


def upload_command(args: argparse.Namespace) -> int:
    plan = build_upload_plan(resolve_cli_path(args.input), args.batch_size)
    print_plan(plan)
    state_path = resolve_cli_path(args.state)
    if state_path.exists() and not args.replace_state:
        raise MaprayUploadError(
            f"state file already exists: {state_path}; use a new path or --replace-state"
        )

    api = MaprayB3dApi(require_token(), timeout=args.timeout)
    attribution = {
        "display": args.attribution,
        "link": args.attribution_link,
        "description": "PLATEAU source CityGML",
    }
    dataset = api.create_dataset(
        name=args.name,
        description=args.description,
        srid=args.srid,
        attributions=[attribution],
    )
    dataset_id = str(dataset["id"])
    state: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_name": args.name,
        "status": "uploading",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "plan": plan.as_dict(),
        "completed_batches": 0,
        "uploaded_files": 0,
    }
    write_state(state_path, state)
    print(f"[CREATED] Building Dataset ID: {dataset_id}")
    print(f"[STATE] {state_path}")

    try:
        for batch_index, batch in enumerate(plan.batches, start=1):
            print(
                f"[BATCH {batch_index}/{len(plan.batches)}] "
                f"requesting {len(batch)} upload URL(s)"
            )
            targets = api.create_upload_urls(dataset_id, batch)
            last_printed = 0

            def progress(done: int, total: int) -> None:
                nonlocal last_printed
                if done == total or done - last_printed >= 50:
                    print(f"  uploaded {done}/{total}")
                    last_printed = done

            upload_batch(
                batch,
                targets,
                workers=args.workers,
                retries=args.retries,
                timeout=args.timeout,
                on_progress=progress,
            )
            state["completed_batches"] = batch_index
            state["uploaded_files"] = int(state["uploaded_files"]) + len(batch)
            state["updated_at"] = utc_now()
            write_state(state_path, state)

        print("[CONVERT] all files uploaded; starting B3D conversion")
        api.convert(dataset_id)
        state["status"] = "conversion_requested"
        state["updated_at"] = utc_now()
        write_state(state_path, state)

        if args.wait:
            deadline = time.monotonic() + args.wait_timeout
            while True:
                dataset_info = api.dataset(dataset_id)
                status = str(dataset_info.get("status", "unknown"))
                error = str(dataset_info.get("error", ""))
                print(f"[STATUS] {status}" + (f": {error}" if error else ""))
                state["status"] = status
                state["updated_at"] = utc_now()
                if error:
                    state["error"] = error
                write_state(state_path, state)
                if status == "ready":
                    break
                if status in {"error", "failed"} or error:
                    raise MaprayUploadError(f"B3D conversion failed: {error or status}")
                if time.monotonic() >= deadline:
                    raise MaprayUploadError(
                        f"timed out waiting for conversion; Dataset ID remains {dataset_id}"
                    )
                time.sleep(args.poll_interval)

        print(f"[OK] Building Dataset ID: {dataset_id}")
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["updated_at"] = utc_now()
        state["error"] = str(error)
        write_state(state_path, state)
        print(f"[FAILED DATASET] {dataset_id}", file=sys.stderr)
        raise


def status_command(args: argparse.Namespace) -> int:
    api = MaprayB3dApi(require_token(), timeout=args.timeout)
    dataset = api.dataset(args.dataset_id)
    safe = {
        key: dataset.get(key)
        for key in ("id", "name", "srid", "status", "error", "url", "created_at", "updated_at")
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="validate files and print batches")
    plan_parser.add_argument("--input", required=True, help="directory containing GML and textures")
    plan_parser.add_argument("--batch-size", type=int, default=1000)
    plan_parser.set_defaults(handler=plan_command)

    upload_parser = subparsers.add_parser("upload", help="create, upload, and convert a B3D dataset")
    upload_parser.add_argument("--input", required=True, help="directory containing GML and textures")
    upload_parser.add_argument("--name", default="shibuya-textured-2023")
    upload_parser.add_argument(
        "--description", default="PLATEAU Shibuya CityGML with referenced textures"
    )
    upload_parser.add_argument("--srid", type=int, default=4326)
    upload_parser.add_argument("--attribution", default=DEFAULT_ATTRIBUTION)
    upload_parser.add_argument("--attribution-link", default=DEFAULT_ATTRIBUTION_LINK)
    upload_parser.add_argument("--batch-size", type=int, default=1000)
    upload_parser.add_argument("--workers", type=int, default=8)
    upload_parser.add_argument("--retries", type=int, default=3)
    upload_parser.add_argument("--timeout", type=float, default=120.0)
    upload_parser.add_argument(
        "--state",
        default=str(
            WORKSPACE_ROOT
            / "runtime"
            / "windows"
            / "generated"
            / "shibuya"
            / "mapray-b3d-upload-state.json"
        ),
    )
    upload_parser.add_argument("--replace-state", action="store_true")
    upload_parser.add_argument("--wait", action="store_true")
    upload_parser.add_argument("--wait-timeout", type=float, default=7200.0)
    upload_parser.add_argument("--poll-interval", type=float, default=15.0)
    upload_parser.set_defaults(handler=upload_command)

    status_parser = subparsers.add_parser("status", help="retrieve a B3D dataset status")
    status_parser.add_argument("--dataset-id", required=True)
    status_parser.add_argument("--timeout", type=float, default=60.0)
    status_parser.set_defaults(handler=status_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
