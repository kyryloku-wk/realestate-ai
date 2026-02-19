# realestateai/data/datasets/dataset_processor.py
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import dotenv
import fsspec
import pandas as pd

from realestateai.data.postgres.utils import query_to_dataframe

dotenv.load_dotenv()

QUERY_TO_LOAD_DATASET = """
SELECT * FROM listings_silver
"""

DEFAULT_DATASETS_LOCATION = "s3://realestate/train_datasets"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_remote(path: str) -> bool:
    return "://" in path


def _ensure_local_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _df_schema(df: pd.DataFrame) -> dict[str, str]:
    return {str(c): str(df[c].dtype) for c in df.columns}


@dataclass(frozen=True)
class DatasetVersionInfo:
    name: str
    version: str
    parquet_path: str
    metadata_path: str
    created_at: str
    row_count: int
    schema: dict[str, str]
    query: str


class DatasetProcessor:
    """
    Versioned dataset storage in Parquet + metadata.json.

    Layout:
      {base}/{name}/
        latest.json
        versions/{version}/
          data.parquet
          metadata.json

    Important:
    - For MinIO/S3 always provide storage_options (s3fs / fsspec kwargs).
    - We write/read parquet through fsspec file objects to avoid pyarrow-native S3.
    """

    def __init__(
        self,
        main_s3_folder: str = DEFAULT_DATASETS_LOCATION,
        storage_options: dict[str, Any] | None = None,  # use minio storage creads lower
        parquet_filename: str = "data.parquet",
        versions_dirname: str = "versions",
    ) -> None:
        self.base = main_s3_folder.rstrip("/")
        self.storage_options = storage_options or {}
        self.parquet_filename = parquet_filename
        self.versions_dirname = versions_dirname

    def _dataset_root(self, name: str) -> str:
        return f"{self.base}/{name}"

    def _latest_path(self, name: str) -> str:
        return f"{self._dataset_root(name)}/latest.json"

    def _version_root(self, name: str, version: str) -> str:
        return f"{self._dataset_root(name)}/{self.versions_dirname}/{version}"

    def _parquet_path(self, name: str, version: str) -> str:
        return f"{self._version_root(name, version)}/{self.parquet_filename}"

    def _metadata_path(self, name: str, version: str) -> str:
        return f"{self._version_root(name, version)}/metadata.json"

    def _write_text(self, path: str, text: str) -> None:
        if _is_remote(path):
            with fsspec.open(path, "w", **self.storage_options) as f:
                f.write(text)
        else:
            _ensure_local_dir(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

    def _read_text(self, path: str) -> str:
        if _is_remote(path):
            with fsspec.open(path, "r", **self.storage_options) as f:
                return f.read()
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _exists(self, path: str) -> bool:
        if _is_remote(path):
            fs, fs_path = fsspec.core.url_to_fs(path, **self.storage_options)
            return fs.exists(fs_path)
        return os.path.exists(path)

    def create_new_dataset_version(
        self,
        name: str = "default",
        df: pd.DataFrame | None = None,
        query: str = QUERY_TO_LOAD_DATASET,
        version: str | None = None,
        parquet_kwargs: dict[str, Any] | None = None,
    ) -> DatasetVersionInfo:
        if df is None:
            df = query_to_dataframe(query)

        created_at = _utc_now_iso()
        if version is None:
            version = f"{created_at.replace(':','').replace('-','')}-{uuid.uuid4().hex[:8]}"

        parquet_path = self._parquet_path(name, version)
        meta_path = self._metadata_path(name, version)

        parquet_kwargs = parquet_kwargs or {}
        parquet_kwargs.setdefault("engine", "pyarrow")
        parquet_kwargs.setdefault("index", False)

        # ✅ CRITICAL: write via fsspec file object for remote paths
        if _is_remote(parquet_path):
            with fsspec.open(parquet_path, "wb", **self.storage_options) as f:
                df.to_parquet(f, **parquet_kwargs)
        else:
            _ensure_local_dir(parquet_path)
            df.to_parquet(parquet_path, **parquet_kwargs)

        info = DatasetVersionInfo(
            name=name,
            version=version,
            parquet_path=parquet_path,
            metadata_path=meta_path,
            created_at=created_at,
            row_count=int(len(df)),
            schema=_df_schema(df),
            query=query,
        )

        meta_obj = {
            "name": info.name,
            "version": info.version,
            "created_at": info.created_at,
            "row_count": info.row_count,
            "schema": info.schema,
            "query": info.query,
            "parquet_path": info.parquet_path,
        }
        self._write_text(meta_path, json.dumps(meta_obj, ensure_ascii=False, indent=2))

        latest_obj = {"name": name, "latest_version": version, "updated_at": created_at}
        self._write_text(
            self._latest_path(name), json.dumps(latest_obj, ensure_ascii=False, indent=2)
        )

        return info

    def get_latest_dataset(self, name: str = "default") -> tuple[pd.DataFrame, DatasetVersionInfo]:
        latest_path = self._latest_path(name)
        if not self._exists(latest_path):
            raise FileNotFoundError(f"latest.json not found: {latest_path}")

        latest_obj = json.loads(self._read_text(latest_path))
        version = latest_obj["latest_version"]
        return self.get_dataset_by_version(name=name, version=version)

    def get_dataset_by_version(
        self,
        name: str = "default",
        version: str = "",
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, DatasetVersionInfo]:
        if not version:
            raise ValueError("version must be provided")

        parquet_path = self._parquet_path(name, version)
        meta_path = self._metadata_path(name, version)

        if not self._exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        if not self._exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        meta_obj = json.loads(self._read_text(meta_path))

        # ✅ CRITICAL: read via fsspec file object for remote paths
        if _is_remote(parquet_path):
            with fsspec.open(parquet_path, "rb", **self.storage_options) as f:
                df = pd.read_parquet(f, engine="pyarrow", columns=columns)
        else:
            df = pd.read_parquet(parquet_path, engine="pyarrow", columns=columns)

        info = DatasetVersionInfo(
            name=meta_obj["name"],
            version=meta_obj["version"],
            parquet_path=meta_obj["parquet_path"],
            metadata_path=meta_path,
            created_at=meta_obj["created_at"],
            row_count=int(meta_obj["row_count"]),
            schema=dict(meta_obj["schema"]),
            query=str(meta_obj.get("query", "")),
        )
        return df, info


# -----------------------------
# MinIO helper (matches your working s3fs config)
# -----------------------------
def minio_storage_options_from_env() -> dict[str, Any]:
    return {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "client_kwargs": {
            "endpoint_url": os.environ["S3_ENDPOINT"],
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        },
        "config_kwargs": {
            "signature_version": "s3v4",
            "s3": {"addressing_style": "path"},
        },
    }


if __name__ == "__main__":
    DatasetProcessor(storage_options=minio_storage_options_from_env()).create_new_dataset_version()
    df, info = DatasetProcessor(
        storage_options=minio_storage_options_from_env()
    ).get_latest_dataset()
    print(df.head())
    print(info)
