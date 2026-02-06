import json
import time
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from realestateai.data.minio import get_bucket_name, upload_html
from realestateai.data.postgres.listing_table import HtmlFileCreate, save_scraping_metadata

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


def get_listing_urls(
    page: int = 1, headers: dict | None = HEADERS, search_url: str | None = SEARCH_URL
):
    """
    Возвращает список объявлений с выдачи (organic):
      {
        "ad_id": int|None,      # из tracking.listing.ad_impressions (если нашли)
        "url": str,
        "title": str,
        "address": str
      }

    Связка ad_id -> карточка делается ПО ПОРЯДКУ (index-based).
    """
    if headers is None:
        headers = {}

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

    offers = []
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

        offers.append(
            {
                "ad_id": ad_id,
                "offer_id": f"{title}_{address}_{ad_id or 'noad'}",  # для удобства, можно потом убрать
                "url": full_url,
                "title": title,
                "address": address,
            }
        )

    return offers


def save_listings_html(pages: int = 1, delay: float = 1.0, search_url: str = SEARCH_URL):
    """Fetch each listing HTML, upload it to MinIO and save metadata to Postgres."""
    bucket = get_bucket_name()
    for page in range(1, pages + 1):
        offers = get_listing_urls(page=page, search_url=search_url)
        print(f"Found {len(offers)} offers on page {page}")
        for offer in offers:
            offer_id = offer["offer_id"]
            url = offer["url"]
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    print(f"Failed to fetch {url}: {r.status_code}")
                    continue

                # sha = hashlib.sha256(url.encode()).hexdigest()
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                key = f"listings/{ts}_{offer_id}.html"

                upload_html(bucket, key, r.content)

                try:
                    new_id = save_scraping_metadata(
                        HtmlFileCreate(
                            offer_id=offer_id,
                            url=url,
                            minio_key=key,
                            scraped_at=datetime.utcnow(),
                            address=offer.get("address"),
                            title=offer.get("title"),
                            ad_id=offer.get("ad_id"),
                        )
                    )
                    print(f"Saved {url} -> s3://{bucket}/{key}, db_id={new_id}")
                    yield new_id
                except Exception as ex:
                    print(f"Uploaded to MinIO but failed to save metadata for {url}: {ex}")

            except Exception as e:
                print(f"Error processing {url}: {e}")

            time.sleep(delay)
            # break


if __name__ == "__main__":
    print(get_listing_urls(page=1, search_url=SEARCH_URL))
    # list(save_listings_html(pages=1, delay=2.0))
