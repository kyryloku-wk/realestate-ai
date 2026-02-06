from random import random

from tqdm import tqdm

from realestateai.data.load_htmls import save_listings_html
from realestateai.data.minio import get_bucket_name, load_html
from realestateai.data.postgres.bronze_ingestion import load_listing_raw, upsert_listing_raw
from realestateai.data.postgres.db import init_db
from realestateai.data.postgres.listing_table import load_html_file
from realestateai.data.scrapers.new_hard_scrapper import parse_otodom_html_v2

if __name__ == "__main__":
    init_db()

    SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?limit=72&ownerTypeSingleSelect=ALL&areaMin=55&priceMax=1200000&roomsNumber=%5BTHREE%2CFOUR%2CFIVE%2CSIX_OR_MORE%5D&by=LATEST&direction=DESC"
    # Получаем HTML и сохраняем в MinIO + Postgres
    max_items = 3600
    curr_item = 0
    max_repeated_ad_ids = 5
    for row_id in tqdm(
        save_listings_html(pages=99999, delay=1 + random() * 2, search_url=SEARCH_URL),
        total=max_items,
    ):
        listing_row = load_html_file(row_id=row_id)
        minio_key = listing_row.minio_key if listing_row else None
        html = load_html(get_bucket_name(), minio_key).decode("utf-8", errors="replace")

        result = parse_otodom_html_v2(html)
        if load_listing_raw(ad_id=result.get("ad_id")):
            print(f"Ad ID {result.get('ad_id')} already exists in listings_raw. Skipping.")
            max_repeated_ad_ids -= 1
            if max_repeated_ad_ids <= 0:
                print("Reached maximum repeated ad IDs. Stopping.")
                break
            continue

        row_id = upsert_listing_raw(result)
        res = load_listing_raw(row_id=row_id)  # just to verify it was saved correctly:
        max_items -= 1
        if max_items <= 0:
            print("Reached maximum items limit. Stopping.")
            break
