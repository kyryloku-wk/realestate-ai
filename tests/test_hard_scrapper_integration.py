import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# Helpers to load modules by file path (works even if package name has hyphens)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "src" / "realestateai" / "data"


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


postgres = load_module_from_path("postgres_mod", DATA_DIR / "postgres.py")
minio = load_module_from_path("minio_mod", DATA_DIR / "minio.py")
hard_scrapper = load_module_from_path("hard_scrapper_mod", DATA_DIR / "hard_scrapper.py")


def fake_geocode(address: str):
    return SimpleNamespace(latitude=50.061, longitude=19.938)


@pytest.mark.integration
def test_parse_listings_from_minio_and_db(monkeypatch):
    """Integration test: query html_files, download HTML from MinIO and parse it."""

    monkeypatch.setattr(hard_scrapper.geolocator, "geocode", fake_geocode)

    engine = getattr(postgres, "engine", None)
    html_files = getattr(postgres, "html_files", None)

    if engine is None or html_files is None:
        pytest.skip("Postgres not configured (DATABASE_URL missing or engine not available)")

    # Use helper to run SQL and return a pandas DataFrame
    df = postgres.query_to_dataframe("SELECT * FROM html_files ORDER BY id DESC LIMIT 2")
    if df.empty:
        pytest.skip("No rows found in html_files table - run scraping first")

    # Convert dataframe rows to dicts for compatibility with the rest of the test
    rows = df.to_dict(orient="records")

    bucket = minio.get_bucket_name()

    any_good = False
    errors = []

    for row in rows:
        key = row["minio_key"]
        offer_id = row["offer_id"]

        try:
            html_bytes = minio.load_html(bucket, key)
        except Exception as e:
            pytest.skip(f"Cannot load object s3://{bucket}/{key} from MinIO: {e}")

        html_text = html_bytes.decode("utf-8", errors="replace")

        parsed = hard_scrapper.parse_html(html_text)

        assert isinstance(parsed, dict)

        # Basic expectations: description must be present and non-empty for a valid listing
        desc = parsed.get("description")
        if desc and isinstance(desc, str) and desc.strip():
            any_good = True
        else:
            errors.append(f"Missing or empty description for offer {offer_id} (key={key})")

        # If price is present, it should be an int > 0
        if "price" in parsed:
            assert isinstance(parsed["price"], int) and parsed["price"] > 0

        # If area is present, it should be a float
        if "area" in parsed:
            assert isinstance(parsed["area"], float)

    assert (
        any_good
    ), "None of the fetched listings had a non-empty description. Details: " + "; ".join(errors)
