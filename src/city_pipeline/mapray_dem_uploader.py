"""Mapray Cloud DEM Dataset API client."""

from __future__ import annotations

from pathlib import Path

from mapray_b3d_uploader import (
    MaprayB3dApi,
    MaprayUploadError,
    UploadFile,
    upload_with_retry,
)


class MaprayDemApi(MaprayB3dApi):
    def create_dem_dataset(
        self,
        name: str,
        description: str,
        attributions: list[dict[str, str]],
    ) -> dict[str, object]:
        result = self._json_request(
            "POST",
            "/demdatasets/v2",
            {
                "name": name,
                "description": description,
                "attributions": attributions,
                "src_file_type": "geotiff",
                "dst_file_type": "dem",
            },
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise MaprayUploadError("Create DEM Dataset did not return a dataset ID")
        return result

    def create_dem_upload_url(
        self, dataset_id: str, item: UploadFile
    ) -> dict[str, object]:
        result = self._json_request(
            "POST", f"/demdatasets/v2/uploads/{dataset_id}", [item.as_request()]
        )
        if not isinstance(result, list) or len(result) != 1:
            raise MaprayUploadError("Create DEM upload URL returned invalid data")
        return result[0]

    def convert_dem(self, dataset_id: str) -> object:
        return self._json_request("POST", f"/demdatasets/v2/convert/{dataset_id}")

    def dem_dataset(self, dataset_id: str) -> dict[str, object]:
        result = self._json_request("GET", f"/demdatasets/v2/{dataset_id}")
        if not isinstance(result, dict):
            raise MaprayUploadError("Retrieve DEM Dataset returned invalid data")
        return result


def upload_geotiff(
    api: MaprayDemApi,
    dataset_id: str,
    path: Path,
    retries: int = 3,
    timeout: float = 120.0,
) -> UploadFile:
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
        raise MaprayUploadError(f"DEM input must be a GeoTIFF file: {path}")
    item = UploadFile(path=path, name=path.name, size=path.stat().st_size)
    target = api.create_dem_upload_url(dataset_id, item)
    upload_with_retry(item, target, retries=retries, timeout=timeout)
    return item
