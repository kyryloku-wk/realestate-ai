from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _safe_get(obj: Any, path: list[Any], default=None):
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
    m = _INT_RE.search(str(text).replace("\xa0", " "))
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _parse_float(text: str | None) -> float | None:
    if not text:
        return None
    s = str(text).replace("\xa0", " ").strip()
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
    return re.sub(r"\s+", " ", str(text)).strip() or None


def _extract_next_data(soup: BeautifulSoup) -> dict | None:
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except Exception:
        return None


def _info_list_to_kv(items: Any) -> dict[str, dict[str, Any]]:
    """
    Convert Otodom AdditionalInfo[] like:
      {label, values: [...], unit: "..."}
    into:
      {label: {"values":[...], "unit":"..."}}
    (keeps values as-is, because new labels may appear)
    """
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        label = it.get("label")
        if not label:
            continue
        out[label] = {
            "values": it.get("values") if isinstance(it.get("values"), list) else [],
        }
        if out[label].get("unit"):
            out[label]["unit"] = it.get("unit")
    return out


def _characteristics_to_map(chars: Any) -> dict[str, dict[str, Any]]:
    """
    characteristics[] -> dict:
      key -> {value, localizedValue, currency}
    """
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(chars, list):
        return out
    for ch in chars:
        if not isinstance(ch, dict):
            continue
        k = ch.get("key")
        if not k:
            continue
        out[k] = {
            "value": ch.get("value"),
            "localized": ch.get("localizedValue"),
        }
        if ch.get("currency"):
            out[k]["currency"] = ch.get("currency")
    return out


def _parse_bool_from_info_value(values: list[str]) -> bool | None:
    """
    Otodom sometimes encodes booleans like:
      ["::n"] or ["::y"] in additionalInformation for fields like lift.
    Also can be ["lift::yes"] in some variants.
    """
    joined = " ".join([str(v) for v in values]).lower()
    if "::n" in joined or "no" in joined or "::0" in joined:
        return False
    if "::y" in joined or "yes" in joined or "::1" in joined:
        return True
    return None


def parse_otodom_html_v2(html_text: str, keep_ad_raw: bool = False) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")

    next_data = _extract_next_data(soup)
    ad = _safe_get(next_data, ["props", "pageProps", "ad"]) if next_data else None

    out: dict[str, Any] = {
        "source": "otodom",
        "next_data_present": bool(next_data),
        # Если хочешь хранить next_data для дебага — оставь. Иначе лучше выключить (очень жирный).
        "next_data": None,
        "ad_raw": ad if keep_ad_raw else None,
    }

    if not isinstance(ad, dict):
        return out

    # -------------------------
    # Identifiers / status / timestamps
    # -------------------------
    out["ad_id"] = ad.get("id")
    out["reference_id"] = ad.get("referenceId")
    out["slug"] = ad.get("slug")
    out["url"] = ad.get("url")
    out["status"] = ad.get("status")
    out["created_at"] = ad.get("createdAt")
    out["modified_at"] = ad.get("modifiedAt")
    out["pushed_up_at"] = ad.get("pushedUpAt")

    # -------------------------
    # Title / SEO
    # -------------------------
    out["title"] = ad.get("title")
    seo = ad.get("seo") if isinstance(ad.get("seo"), dict) else {}
    out["seo_title"] = seo.get("title")
    out["seo_description"] = seo.get("description")

    # -------------------------
    # Description (raw HTML + optional plain)
    # -------------------------
    out["description_html"] = ad.get("description")
    # plain text можно позже делать отдельной джобой; но как быстрый вариант:
    out["description_text"] = _clean_text(
        BeautifulSoup(ad.get("description") or "", "html.parser").get_text(" ")
    )

    # -------------------------
    # Category / market / advertiser
    # -------------------------
    out["market"] = ad.get("market")
    out["advertiser_type"] = ad.get("advertiserType")
    out["advert_type"] = ad.get("advertType")
    out["exclusive_offer"] = ad.get("exclusiveOffer")
    out["creation_source"] = ad.get("creationSource")

    # -------------------------
    # Location (coords + address + hierarchy)
    # -------------------------
    loc = ad.get("location") if isinstance(ad.get("location"), dict) else {}
    coords = loc.get("coordinates") if isinstance(loc.get("coordinates"), dict) else {}
    out["latitude"] = coords.get("latitude")
    out["longitude"] = coords.get("longitude")

    addr = loc.get("address") if isinstance(loc.get("address"), dict) else {}

    def _nm(d):  # {"name":...}
        return d.get("name") if isinstance(d, dict) else None

    out["street"] = _nm(addr.get("street"))
    out["street_number"] = _clean_text(_safe_get(addr, ["street", "number"]))
    out["district"] = _nm(addr.get("district"))
    out["city"] = _nm(addr.get("city"))
    out["county"] = _nm(addr.get("county"))
    out["province"] = _nm(addr.get("province"))
    out["postal_code"] = addr.get("postalCode")

    # Full human-readable location string (for logs / debugging / NER etc.)
    loc_parts = [
        p
        for p in [out.get("street"), out.get("district"), out.get("city"), out.get("province")]
        if p
    ]
    out["location_text"] = ", ".join(loc_parts) if loc_parts else None

    # rev = _safe_get(loc, ["reverseGeocoding", "locations"], default=[])
    # if isinstance(rev, list):
    #     out["location_hierarchy"] = [
    #         {
    #             "id": x.get("id"),
    #             "level": x.get("locationLevel"),
    #             "name": x.get("name"),
    #             "full_name": x.get("fullName"),
    #         }
    #         for x in rev
    #         if isinstance(x, dict)
    #     ]
    # else:
    #     out["location_hierarchy"] = []

    # -------------------------
    # Images: only SMALL urls
    # -------------------------
    images = ad.get("images") if isinstance(ad.get("images"), list) else []
    out["images_small"] = [
        img.get("small") for img in images if isinstance(img, dict) and img.get("small")
    ]
    out["images_count"] = len(images)

    # -------------------------
    # Features (lists)
    # -------------------------
    out["features"] = ad.get("features") if isinstance(ad.get("features"), list) else []
    # features by category -> dict label -> values
    fbc = ad.get("featuresByCategory") if isinstance(ad.get("featuresByCategory"), list) else []
    features_by_category: dict[str, list[str]] = {}
    for g in fbc:
        if not isinstance(g, dict):
            continue
        label = g.get("label")
        vals = g.get("values")
        if label and isinstance(vals, list):
            features_by_category[label] = vals
    out["features_by_category"] = features_by_category
    out["features_without_category"] = (
        ad.get("featuresWithoutCategory")
        if isinstance(ad.get("featuresWithoutCategory"), list)
        else []
    )

    # -------------------------
    # Owner / agency
    # -------------------------
    owner = ad.get("owner") if isinstance(ad.get("owner"), dict) else {}
    agency = ad.get("agency") if isinstance(ad.get("agency"), dict) else {}

    out["owner"] = {
        "id": owner.get("id"),
        "name": owner.get("name"),
        "type": owner.get("type"),
        "phones": owner.get("phones") if isinstance(owner.get("phones"), list) else [],
        "imageUrl": owner.get("imageUrl"),
    }

    out["agency_name"] = (
        agency.get("name") if agency else None
    )  # для удобного поиска по агентству в будущем
    # out["agency"] = {
    #     "id": agency.get("id"),
    #     "name": agency.get("name"),
    #     "type": agency.get("type"),
    #     "phones": agency.get("phones") if isinstance(agency.get("phones"), list) else [],
    #     "address": agency.get("address"),
    #     "url": agency.get("url"),
    #     "brandingVisible": agency.get("brandingVisible"),
    #     "enabledFeatures": (
    #         agency.get("enabledFeatures") if isinstance(agency.get("enabledFeatures"), list) else []
    #     ),
    # }

    # -------------------------
    # Characteristics + Info blocks (keep as-is but normalized)
    # -------------------------
    out["char"] = _characteristics_to_map(ad.get("characteristics"))
    out["top_info"] = _info_list_to_kv(ad.get("topInformation"))
    out["additional_info"] = _info_list_to_kv(ad.get("additionalInformation"))

    # -------------------------
    # ML-friendly flattened core features
    # (остальное ты потом сам решишь, но это базовый набор)
    # -------------------------
    char = out["char"]

    def cval(k: str) -> str | None:
        v = char.get(k, {}).get("value")
        return str(v) if v is not None else None

    def cloc(k: str) -> str | None:
        v = char.get(k, {}).get("localized")
        return str(v) if v is not None else None

    out["price_pln"] = _parse_int(cval("price") or cloc("price"))
    out["area_m2"] = _parse_float(cval("m") or cloc("m"))
    out["price_per_m2_pln"] = _parse_int(cval("price_per_m") or cloc("price_per_m"))
    out["rooms"] = _parse_int(cval("rooms_num") or cloc("rooms_num"))
    out["building_floors"] = _parse_int(cval("building_floors_num") or cloc("building_floors_num"))
    out["year_built"] = _parse_int(cval("build_year") or cloc("build_year"))
    out["rent_pln"] = _parse_int(cval("rent") or cloc("rent"))

    # floor_no comes as "floor_3" -> extract number
    floor_raw = cval("floor_no") or cloc("floor_no")
    out["floor"] = _parse_int(floor_raw)

    # lift: в твоем примере он не в characteristics, но есть в additionalInformation:
    # {"label": "lift", "values": ["::n"]}
    lift_values = _safe_get(out, ["additional_info", "lift", "values"], default=[])
    if isinstance(lift_values, list):
        out["lift"] = _parse_bool_from_info_value(lift_values)
    else:
        out["lift"] = None

    # Derived check: compute price_per_m2 if missing
    if out.get("price_pln") and out.get("area_m2") and not out.get("price_per_m2_pln"):
        try:
            out["price_per_m2_pln"] = int(round(out["price_pln"] / float(out["area_m2"])))
        except Exception:
            pass

    # -------------------------
    # Optional: keep "property" block (structured, useful for future)
    # -------------------------
    out["property_raw"] = ad.get("property") if isinstance(ad.get("property"), dict) else None

    return out


if __name__ == "__main__":
    from realestateai.data.minio import get_bucket_name, load_html
    from realestateai.data.postgres.utils import query_to_dataframe

    df = query_to_dataframe("SELECT * FROM html_files ORDER BY id DESC LIMIT 5").iloc[3]

    file_string = load_html(get_bucket_name(), df["minio_key"]).decode("utf-8", errors="replace")

    result = parse_otodom_html_v2(file_string)

    json.dump(
        result,
        fp=open("garbage/parsed_output.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
