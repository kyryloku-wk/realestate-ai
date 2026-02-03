from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

# ----------------------------
# Helpers
# ----------------------------

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _safe_get(obj: Any, path: list[Any], default=None):
    """Safely get nested values from dict/list by path."""
    cur = obj
    for p in path:
        try:
            if isinstance(cur, dict):
                cur = cur.get(p)
            elif isinstance(cur, list) and isinstance(p, int):
                cur = cur[p]
            else:
                return default
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    m = _INT_RE.search(text.replace("\xa0", " "))
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _parse_float_pl(text: str | None) -> float | None:
    """Parses floats like '68,71 m²' or '15267' or '15 267 zł/m²' -> 68.71 / 15267.0."""
    if not text:
        return None
    s = text.replace("\xa0", " ").strip()
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def _clean_text(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    return t or None


def _extract_meta(soup: BeautifulSoup) -> dict[str, str | None]:
    def meta(name: str = None, prop: str = None) -> str | None:
        if name:
            tag = soup.find("meta", attrs={"name": name})
            return tag.get("content") if tag else None
        if prop:
            tag = soup.find("meta", attrs={"property": prop})
            return tag.get("content") if tag else None
        return None

    canonical = soup.find("link", rel="canonical")
    return {
        "meta_title": _clean_text(soup.title.text) if soup.title else None,
        "meta_description": _clean_text(meta(name="description")),
        "og_title": _clean_text(meta(prop="og:title")),
        "og_description": _clean_text(meta(prop="og:description")),
        "og_image": meta(prop="og:image"),
        "canonical_url": canonical.get("href") if canonical else None,
    }


def _extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    crumbs = []
    for a in soup.select('a[data-cy="breadcrumb"]'):
        txt = _clean_text(a.get_text())
        if txt:
            crumbs.append(txt)
    return crumbs


def _extract_next_data(soup: BeautifulSoup) -> dict | None:
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except Exception:
        return None


# ----------------------------
# Main parser
# ----------------------------


def parse_otodom_html(html_text: str) -> dict[str, Any]:
    """
    Parse a single Otodom offer HTML page into a dict of useful fields.

    Strategy:
      1) Prefer __NEXT_DATA__ JSON (stable, rich).
      2) Fallback to meta tags / breadcrumbs / simple DOM selectors.

    Returns:
      dict with both:
        - top-level normalized fields (good for ML features)
        - raw blocks: next_data, ad_raw, characteristics_raw, images_raw
    """
    soup = BeautifulSoup(html_text, "html.parser")

    out: dict[str, Any] = {
        "source": "otodom",
    }

    # --- Meta / fallback info
    meta = _extract_meta(soup)
    out.update(meta)
    out["breadcrumbs"] = _extract_breadcrumbs(soup)

    # Sometimes title is easy to grab from DOM too
    dom_title = soup.find(attrs={"data-cy": "adPageAdTitle"})
    if dom_title:
        out["title_dom"] = _clean_text(dom_title.get_text())

    # --- NEXT_DATA (main truth)
    next_data = _extract_next_data(soup)
    out["next_data_present"] = bool(next_data)
    out["next_data"] = None  # keep optional; can be large (you can drop it later)

    ad = None
    if next_data:
        out["next_data"] = None  # if you don't want to store it, set to None
        ad = _safe_get(next_data, ["props", "pageProps", "ad"])

    out["ad_raw"] = ad  # raw ad block (can be big; you can drop later)

    # --- Basic identifiers / timestamps
    out["ad_id"] = _safe_get(ad, ["id"])
    out["public_id"] = _safe_get(ad, ["publicId"])
    out["slug"] = _safe_get(ad, ["slug"])
    out["created_at"] = _safe_get(ad, ["createdAt"])
    out["modified_at"] = _safe_get(ad, ["modifiedAt"])

    # URL: canonical usually full; slug alone can be used to build URL
    out["url"] = out.get("canonical_url")
    if not out["url"] and out.get("slug"):
        # best-effort
        out["url"] = f"https://www.otodom.pl/pl/oferta/{out['slug']}"

    # --- Market / seller
    out["market"] = _safe_get(ad, ["market"])  # e.g. SECONDARY
    out["advertiser_type"] = _safe_get(ad, ["advertiserType"])  # business/private...
    out["advert_type"] = _safe_get(ad, ["advertType"])  # AGENCY etc.
    out["source_urn"] = _safe_get(ad, ["source"])

    # --- Description (often in JSON)
    out["description"] = _clean_text(_safe_get(ad, ["description"]))

    # --- Location details
    coords = _safe_get(ad, ["location", "coordinates"]) or {}
    out["latitude"] = coords.get("latitude")
    out["longitude"] = coords.get("longitude")

    address = _safe_get(ad, ["location", "address"]) or {}
    street = address.get("street") or {}
    district = address.get("district") or {}
    city = address.get("city") or {}
    county = address.get("county") or {}
    province = address.get("province") or {}

    out["street_name"] = street.get("name")
    out["street_number"] = street.get("number") or None
    out["district"] = district.get("name")
    out["city"] = city.get("name")
    out["county"] = county.get("name")
    out["province"] = province.get("name")
    out["postal_code"] = address.get("postalCode")

    # Full human-readable location string (for logs / debugging / NER etc.)
    loc_parts = [
        p
        for p in [out.get("street_name"), out.get("district"), out.get("city"), out.get("province")]
        if p
    ]
    out["location_text"] = ", ".join(loc_parts) if loc_parts else None

    # # Reverse geocoding (often contains hierarchical ids)
    # rev_locations = _safe_get(ad, ["location", "reverseGeocoding", "locations"]) or []
    # out["location_hierarchy"] = [
    #     {
    #         "id": x.get("id"),
    #         "level": x.get("locationLevel"),
    #         "name": x.get("name"),
    #         "full_name": x.get("fullName"),
    #     }
    #     for x in rev_locations
    #     if isinstance(x, dict)
    # ]

    # --- Images
    images = _safe_get(ad, ["images"]) or []
    # out["images_raw"] = images
    # keep multiple sizes for later
    out["image_urls"] = [
        # "thumbnail": [
        #     img.get("thumbnail") for img in images if isinstance(img, dict) and img.get("thumbnail")
        # ],
        img.get("small")
        for img in images
        if isinstance(img, dict) and img.get("small")
        # "medium": [
        #     img.get("medium") for img in images if isinstance(img, dict) and img.get("medium")
        # ],
        # "large": [img.get("large") for img in images if isinstance(img, dict) and img.get("large")],
    ]
    out["images_count"] = len(images)

    # --- Characteristics (ключевая часть для ML)
    characteristics = _safe_get(ad, ["characteristics"]) or []
    # out["characteristics_raw"] = characteristics

    value_map: dict[str, Any] = {}
    localized_map: dict[str, Any] = {}
    currency_map: dict[str, Any] = {}
    for ch in characteristics:
        if not isinstance(ch, dict):
            continue
        k = ch.get("key")
        if not k:
            continue
        value_map[k] = ch.get("value")
        localized_map[k] = ch.get("localizedValue")
        currency_map[k] = ch.get("currency")

    out["char_value"] = value_map
    out["char_localized"] = localized_map
    out["char_currency"] = currency_map

    # Normalize the most useful ML-ready features (stable keys from JSON)
    # (добавляй сюда по мере накопления данных — ключи обычно одинаковые)
    def val(k: str) -> str | None:
        v = value_map.get(k)
        return str(v) if v is not None else None

    def loc(k: str) -> str | None:
        v = localized_map.get(k)
        return str(v) if v is not None else None

    def cur(k: str) -> str | None:
        v = currency_map.get(k)
        return str(v) if v is not None else None

    # Prices
    out["price_pln"] = _parse_int(
        val("price") or loc("price") or out.get("og_description") or out.get("meta_description")
    )
    out["area_m2"] = _parse_float_pl(val("m") or loc("m"))
    out["price_per_m2_pln"] = _parse_int(val("price_per_m") or loc("price_per_m"))

    # Core apartment/house features
    out["rooms"] = _parse_int(val("rooms_num") or loc("rooms_num"))
    out["floor"] = _parse_int(val("floor_no") or loc("floor_no"))
    out["building_floors"] = _parse_int(val("building_floors_num") or loc("building_floors_num"))
    out["year_built"] = _parse_int(val("build_year") or loc("build_year"))

    out["windows_type"] = val("windows_type")
    out["currency"] = cur("price")

    # Categorical (keep both raw code + localized label)
    out["market_type"] = val("market") or loc("market")  # e.g. secondary
    out["building_type"] = val("building_type")
    out["building_type_label"] = loc("building_type")
    out["heating_type"] = val("heating")
    out["heating_type_label"] = loc("heating")
    out["construction_status"] = val("construction_status")
    out["construction_status_label"] = loc("construction_status")
    out["ownership"] = val("building_ownership")
    out["ownership_label"] = loc("building_ownership")

    # Rent / czynsz
    out["rent_pln"] = _parse_int(val("rent") or loc("rent"))

    # --- Derived / quality checks (useful for data validation)
    if out.get("price_pln") and out.get("area_m2") and not out.get("price_per_m2_pln"):
        try:
            out["price_per_m2_pln"] = int(round(out["price_pln"] / float(out["area_m2"])))
        except Exception:
            pass

    out["has_coordinates"] = out.get("latitude") is not None and out.get("longitude") is not None
    out["has_description"] = bool(out.get("description"))

    # del out["next_data"]
    # del out["ad_raw"]

    return out


if __name__ == "__main__":
    from minio import get_bucket_name, load_html
    from postgres import query_to_dataframe

    df = query_to_dataframe("SELECT * FROM html_files ORDER BY id DESC LIMIT 5").iloc[3]

    file_string = load_html(get_bucket_name(), df["minio_key"]).decode("utf-8", errors="replace")

    result = parse_otodom_html(file_string)

    json.dump(
        result, fp=open("parsed_output.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2
    )
