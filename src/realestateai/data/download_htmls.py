import json
import re
import unicodedata
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from realestateai.data.minio import get_bucket_name, upload_html

load_dotenv()

# SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?distanceRadius=75&limit=72&ownerTypeSingleSelect=ALL&areaMin=50&by=LATEST&direction=DESC"  # можно добавить параметры
# SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?distanceRadius=75&by=LATEST&direction=DESC"
# SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?limit=72&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"

SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?limit=72&ownerTypeSingleSelect=ALL&areaMin=55&priceMax=1200000&roomsNumber=%5BTHREE%2CFOUR%2CFIVE%2CSIX_OR_MORE%5D&by=LATEST&direction=DESC"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}


BASE_URL = "https://www.otodom.pl"


class ListingSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ad_id: int | None = None
    offer_id: str = Field(..., min_length=1)
    url: str | None = None
    title: str | None = None
    address: str | None = None


class ListingParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_count: int = 0
    result_count: int = 0
    results_per_page: int = 0


def get_listing_params(
    search_url: str | None = SEARCH_URL,
    headers: dict | None = HEADERS,
) -> ListingParams | None:
    r = requests.get(search_url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    next_script = soup.find("script", id="__NEXT_DATA__")
    if next_script and next_script.string:
        next_data = json.loads(next_script.string)
        listings = (
            next_data.get("props", {}).get("pageProps", {}).get("tracking", {}).get("listing", {})
        )
        page_count = listings.get("page_count", 0)
        result_count = listings.get("result_count", 0)
        results_per_page = listings.get("results_per_page", 0)
        return ListingParams(
            page_count=page_count,
            result_count=result_count,
            results_per_page=results_per_page,
        )
    else:
        return None


def get_urls(
    pages_count,
    search_url: str | None = SEARCH_URL,
    headers: dict | None = HEADERS,
    start_page: int = 1,
):
    for page in range(1, pages_count + 1):
        url = f"{search_url}?page={page}" if "?" not in search_url else f"{search_url}&page={page}"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # ---- 1) Достаём ad_impressions из __NEXT_DATA__ ----
        ad_impressions: list[int] = []
        next_script = soup.find("script", id="__NEXT_DATA__")
        if next_script and next_script.string:
            try:
                next_data = json.loads(next_script.string)
                ad_impressions = (
                    next_data.get("props", {})
                    .get("pageProps", {})
                    .get("tracking", {})
                    .get("listing", {})
                    .get("ad_impressions", [])
                )
                # гарантируем ints
                ad_impressions = [
                    int(x) for x in ad_impressions if isinstance(x, (int, str)) and str(x).isdigit()
                ]
            except Exception:
                ad_impressions = []

        # ---- 2) Парсим карточки из organic выдачи ----
        section = soup.find("div", {"data-cy": "search.listing.organic"})
        if not section:
            return []

        cards = section.select('article[data-sentry-component="AdvertCard"]')

        for idx, card in enumerate(cards):
            link_tag = card.select_one('a[data-cy="listing-item-link"][href]')
            title_tag = card.select_one('p[data-cy="listing-item-title"]')
            if not link_tag or not title_tag:
                continue

            href = link_tag.get("href", "").strip()
            full_url = urljoin(BASE_URL, href)

            title = title_tag.get_text(" ", strip=True)

            address_tag = card.select_one('[data-sentry-component="Address"]')
            address = address_tag.get_text(" ", strip=True) if address_tag else ""

            ad_id = ad_impressions[idx] if idx < len(ad_impressions) else None
            obj = ListingSummary(
                ad_id=ad_id,
                offer_id=f"{title}_{address}",  # для удобства, можно потом убрать
                url=full_url,
                title=title,
                address=address,
            )
            yield obj


def load_and_save_html_to_minio(
    listing_obj: ListingSummary, HEADERS=HEADERS, timeout=15
) -> str | None:
    url = listing_obj.url
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code != 200:
        print(f"Failed to fetch {url}: {r.status_code}")
        return None

    def safe_s3_key(name: str, max_len: int = 200) -> str:
        if not name:
            return "object"
        # Normalize unicode to NFKD and drop non-ascii
        nfkd = unicodedata.normalize("NFKD", str(name))
        ascii_bytes = nfkd.encode("ascii", "ignore")
        ascii_str = ascii_bytes.decode("ascii")
        # Replace whitespace with underscore
        s = re.sub(r"\s+", "_", ascii_str)
        # Allow only a safe subset of characters
        s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
        # Collapse multiple underscores and trim
        s = re.sub(r"_+", "_", s).strip("_")
        if not s:
            s = "object"
        if len(s) > max_len:
            s = s[:max_len]
        return s

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    offer_id_raw = (
        listing_obj.offer_id
        if getattr(listing_obj, "offer_id", None) is not None
        else str(getattr(listing_obj, "ad_id", "unknown"))
    )
    safe_offer_id = safe_s3_key(offer_id_raw)
    key = f"listings/{ts}_{safe_offer_id}.html"

    upload_html(get_bucket_name(), key, r.content)
    return key


if __name__ == "__main__":
    print(get_listing_params(search_url=SEARCH_URL))
    listing_obj = None
    for i, elem in enumerate(get_urls(pages_count=1, search_url=SEARCH_URL)):
        print(elem)
        listing_obj = elem
        if i == 3:
            break

    res = load_and_save_html_to_minio(listing_obj)
    print(res)
