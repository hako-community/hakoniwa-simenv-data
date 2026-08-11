"""Mapray Cloud B3D upload planning and API client.

The Cloud API limits one upload-URL request to 1,000 files.  This module keeps
all batches in one B3D dataset and starts conversion only after every file PUT
has succeeded.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable


API_ROOT = "https://api.mapray.com"
SUPPORTED_SUFFIXES = {".gml", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_URI_RE = re.compile(
    r"<(?:[A-Za-z_][\w.-]*:)?imageURI>([^<]+)</(?:[A-Za-z_][\w.-]*:)?imageURI>"
)


class MaprayUploadError(RuntimeError):
    """Raised for a safe-to-report Mapray upload failure."""


@dataclass(frozen=True)
class UploadFile:
    path: Path
    name: str
    size: int

    def as_request(self) -> dict[str, object]:
        return {"filename": self.name, "size": self.size}


@dataclass(frozen=True)
class UploadPlan:
    root: Path
    files: tuple[UploadFile, ...]
    batches: tuple[tuple[UploadFile, ...], ...]
    gml_count: int
    texture_count: int
    total_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "file_count": len(self.files),
            "gml_count": self.gml_count,
            "texture_count": self.texture_count,
            "total_bytes": self.total_bytes,
            "batch_sizes": [len(batch) for batch in self.batches],
        }


def _read_image_uris(gml_path: Path) -> set[str]:
    uris: set[str] = set()
    with gml_path.open("r", encoding="utf-8") as source:
        for line in source:
            for match in IMAGE_URI_RE.finditer(line):
                value = match.group(1).strip().replace("\\", "/")
                if value:
                    uris.add(value)
    return uris


def _safe_relative_path(root: Path, value: str) -> Path:
    candidate = Path(value.replace("/", os.sep))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise MaprayUploadError(f"unsafe texture path in CityGML: {value}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise MaprayUploadError(f"texture path escapes upload root: {value}") from error
    return resolved


def chunked(values: tuple[UploadFile, ...], size: int) -> tuple[tuple[UploadFile, ...], ...]:
    if size < 1 or size > 1000:
        raise ValueError("batch size must be between 1 and 1000")
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def build_upload_plan(root: Path, batch_size: int = 1000) -> UploadPlan:
    root = root.resolve()
    if not root.is_dir():
        raise MaprayUploadError(f"upload root does not exist: {root}")

    gml_paths = sorted(root.glob("*.gml"))
    if not gml_paths:
        raise MaprayUploadError(f"no CityGML file found directly under: {root}")

    referenced: set[str] = set()
    for gml_path in gml_paths:
        referenced.update(_read_image_uris(gml_path))

    texture_paths: list[Path] = []
    missing: list[str] = []
    for uri in sorted(referenced):
        path = _safe_relative_path(root, uri)
        if not path.is_file():
            missing.append(uri)
        else:
            texture_paths.append(path)
    if missing:
        sample = ", ".join(missing[:5])
        raise MaprayUploadError(
            f"CityGML references {len(missing)} missing texture file(s): {sample}"
        )

    selected_paths = [*gml_paths, *texture_paths]
    upload_files: list[UploadFile] = []
    names: set[str] = set()
    for path in selected_paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise MaprayUploadError(f"unsupported upload file: {path}")
        name = path.relative_to(root).as_posix()
        if name in names:
            continue
        names.add(name)
        upload_files.append(UploadFile(path=path, name=name, size=path.stat().st_size))

    files = tuple(upload_files)
    return UploadPlan(
        root=root,
        files=files,
        batches=chunked(files, batch_size),
        gml_count=len(gml_paths),
        texture_count=len(texture_paths),
        total_bytes=sum(item.size for item in files),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class MaprayB3dApi:
    def __init__(self, token: str, api_root: str = API_ROOT, timeout: float = 60.0):
        if not token.strip():
            raise ValueError("Organization Token is empty")
        self._token = token.strip()
        self._api_root = api_root.rstrip("/")
        self._timeout = timeout

    def _json_request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
    ) -> object:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._api_root + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise MaprayUploadError(
                f"Mapray API {method} {path} failed: HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise MaprayUploadError(f"Mapray API {method} {path} failed: {error.reason}") from error
        if not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise MaprayUploadError(
                f"Mapray API {method} {path} returned invalid JSON"
            ) from error

    def create_dataset(
        self,
        name: str,
        description: str,
        srid: int,
        attributions: list[dict[str, str]],
    ) -> dict[str, object]:
        result = self._json_request(
            "POST",
            "/b3ddatasets/v2",
            {
                "name": name,
                "description": description,
                "attributions": attributions,
                "srid": srid,
            },
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise MaprayUploadError("Create B3D Dataset did not return a dataset ID")
        return result

    def create_upload_urls(
        self,
        dataset_id: str,
        files: Iterable[UploadFile],
    ) -> list[dict[str, object]]:
        payload = [item.as_request() for item in files]
        if len(payload) > 1000:
            raise ValueError("one upload URL request cannot exceed 1000 files")
        result = self._json_request(
            "POST", f"/b3ddatasets/v2/uploads/{dataset_id}", payload
        )
        if not isinstance(result, list) or len(result) != len(payload):
            raise MaprayUploadError(
                f"upload URL count mismatch: requested={len(payload)} returned="
                f"{len(result) if isinstance(result, list) else 'invalid'}"
            )
        return result

    def convert(self, dataset_id: str) -> object:
        return self._json_request("POST", f"/b3ddatasets/v2/convert/{dataset_id}")

    def dataset(self, dataset_id: str) -> dict[str, object]:
        result = self._json_request("GET", f"/b3ddatasets/v2/{dataset_id}")
        if not isinstance(result, dict):
            raise MaprayUploadError("Retrieve B3D Dataset returned invalid data")
        return result

    def conversion_status(self, dataset_id: str) -> dict[str, object]:
        result = self._json_request(
            "GET", f"/b3ddatasets/v2/convert/status/{dataset_id}"
        )
        if not isinstance(result, dict):
            raise MaprayUploadError("Retrieve B3D conversion status returned invalid data")
        return result


def _upload_once(item: UploadFile, target: dict[str, object], timeout: float) -> None:
    url = str(target.get("url", ""))
    if not url:
        raise MaprayUploadError(f"upload URL is missing for {item.name}")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise MaprayUploadError(f"invalid upload URL returned for {item.name}")
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    port = parsed.port
    connection = connection_type(parsed.hostname, port=port, timeout=timeout)
    request_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
    try:
        connection.putrequest("PUT", request_path)
        returned_headers = target.get("headers") or {}
        if not isinstance(returned_headers, dict):
            raise MaprayUploadError(f"invalid upload headers returned for {item.name}")
        lower_headers = {str(key).lower() for key in returned_headers}
        for key, value in returned_headers.items():
            connection.putheader(str(key), str(value))
        if "content-type" not in lower_headers:
            content_type = mimetypes.guess_type(item.name)[0] or "application/octet-stream"
            connection.putheader("Content-Type", content_type)
        connection.putheader("Content-Length", str(item.size))
        connection.endheaders()
        with item.path.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                connection.send(block)
        response = connection.getresponse()
        body = response.read(2048)
        if not 200 <= response.status < 300:
            detail = body.decode("utf-8", errors="replace")
            raise MaprayUploadError(
                f"PUT failed for {item.name}: HTTP {response.status}: {detail}"
            )
    finally:
        connection.close()


def upload_with_retry(
    item: UploadFile,
    target: dict[str, object],
    retries: int,
    timeout: float,
) -> None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _upload_once(item, target, timeout)
            return
        except (OSError, http.client.HTTPException, MaprayUploadError) as error:
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 8))
    raise MaprayUploadError(f"upload failed after retries for {item.name}: {last_error}")


def upload_batch(
    files: tuple[UploadFile, ...],
    targets: list[dict[str, object]],
    workers: int,
    retries: int,
    timeout: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    if len(files) != len(targets):
        raise ValueError("files and upload targets must have the same length")
    target_by_name = {str(target.get("filename")): target for target in targets}
    jobs: list[tuple[UploadFile, dict[str, object]]] = []
    for index, item in enumerate(files):
        target = target_by_name.get(item.name, targets[index])
        jobs.append((item, target))

    completed = 0
    lock = Lock()

    def run(job: tuple[UploadFile, dict[str, object]]) -> None:
        nonlocal completed
        upload_with_retry(job[0], job[1], retries=retries, timeout=timeout)
        with lock:
            completed += 1
            if on_progress:
                on_progress(completed, len(jobs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise

