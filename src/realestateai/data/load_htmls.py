import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from .minio import get_bucket_name, upload_html
    from .postgres import save_scraping_metadata
except Exception:
    # allow running the module directly (no package context)
    from minio import get_bucket_name, upload_html
    from postgres import save_scraping_metadata

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


def get_listing_urls(page: int = 1, headers: dict = HEADERS, search_url: str = SEARCH_URL):
    """Получаем ссылки на объявления с заданной страницы."""
    url = f"{search_url}%3Fpage%3D2&page={page}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    # 1. Находим блок "Wszystkie ogłoszenia"
    all_offers_section = soup.find("div", {"data-cy": "search.listing.organic"})
    if not all_offers_section:
        return []

    offers = []

    # 2. Ищем все <li> (каждое объявление)
    for li in all_offers_section.find_all("li"):
        # 3. Ищем ссылку, title и адрес
        link_tag = li.find("a", {"data-cy": "listing-item-link"}, href=True)
        title_tag = li.find("p", {"data-cy": "listing-item-title"})
        address_tag = li.find("p", class_="css-42r2ms")  # у адреса свой класс

        if link_tag and title_tag:
            href = link_tag["href"]
            full_url = "https://www.otodom.pl" + href if href.startswith("/") else href

            title = title_tag.get_text(strip=True)
            address = address_tag.get_text(strip=True) if address_tag else ""
            id_like = f"{title} | {address}"  # делаем ID из title+address

            offers.append({"id": id_like, "url": full_url, "address": address, "title": title})

    return offers


def save_listings_html(pages: int = 1, delay: float = 1.0):
    """Fetch each listing HTML, upload it to MinIO and save metadata to Postgres."""
    bucket = get_bucket_name()
    for page in range(1, pages + 1):
        offers = get_listing_urls(page=page)
        print(f"Found {len(offers)} offers on page {page}")
        for offer in offers:
            offer_id = offer["id"]
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
                    row_id = save_scraping_metadata(
                        offer_id=offer_id, url=url, minio_key=key, scraped_at=datetime.utcnow()
                    )
                    print(f"Saved {url} -> s3://{bucket}/{key}, db_id={row_id}")
                except Exception as ex:
                    print(f"Uploaded to MinIO but failed to save metadata for {url}: {ex}")

            except Exception as e:
                print(f"Error processing {url}: {e}")

            time.sleep(delay)
            break


if __name__ == "__main__":
    save_listings_html(pages=1, delay=2.0)
