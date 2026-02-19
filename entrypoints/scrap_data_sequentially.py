import logging
import random
import time
from datetime import datetime

from tqdm import tqdm

from realestateai.data.download_htmls import (
    get_listing_params,
    get_urls,
    load_and_save_html_to_minio,
)
from realestateai.data.minio import get_bucket_name, load_html
from realestateai.data.postgres.bronze_ingestion import upsert_broze_parsing
from realestateai.data.postgres.db import init_db
from realestateai.data.postgres.listing_table import (
    HtmlFileCreate,
    get_row_listing_html,
    save_scraping_metadata,
)
from realestateai.data.scrapers.new_hard_scrapper import parse_otodom_html_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MAX_ITEMS = 5000
MAX_PAGES = 70
MAX_REPEATED_IDS = 50
SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?limit=72&ownerTypeSingleSelect=ALL&areaMin=55&priceMax=1200000&roomsNumber=%5BTHREE%2CFOUR%2CFIVE%2CSIX_OR_MORE%5D&by=LATEST&direction=DESC"
START_PAGE = 1

if __name__ == "__main__":
    init_db()

    # Get number of pages and items in the search url
    listing_params = get_listing_params(search_url=SEARCH_URL)
    if listing_params is None:
        logging.error("Failed to get listing parameters.")
        exit(1)

    pages_count = min(listing_params.page_count, MAX_PAGES)
    max_items = min(listing_params.result_count, MAX_ITEMS)
    for i, ad in tqdm(
        enumerate(get_urls(pages_count=pages_count, search_url=SEARCH_URL, start_page=START_PAGE)),
        total=max_items,
    ):
        if i >= max_items:
            break
        # Check if ad_id already exists in listings_raw to avoid duplicates
        listing_row = get_row_listing_html(offer_id=ad.offer_id)
        if listing_row:
            logging.warning(f"Offer ID {ad.offer_id} already exists in html_files. Skipping.")
            MAX_REPEATED_IDS -= 1
            if MAX_REPEATED_IDS <= 0:
                logging.warning("Reached maximum repeated ad IDs. Stopping.")
                break
            continue
        # Load html via requests and save to minio
        minio_key = load_and_save_html_to_minio(ad)

        # Save metadata to html_files table
        db_object = HtmlFileCreate(
            offer_id=ad.offer_id,
            url=ad.url,
            minio_key=minio_key,
            scraped_at=datetime.utcnow(),
            address=ad.address,
            title=ad.title,
            ad_id=ad.ad_id,
        )
        listing_row_id = save_scraping_metadata(db_object)

        # Load the saved html from minio, parse it and save to listings_raw
        file_string = load_html(get_bucket_name(), db_object.minio_key).decode(
            "utf-8", errors="replace"
        )
        parsed_json = parse_otodom_html_v2(file_string)

        bronze_row_id = upsert_broze_parsing(parsed_json)

        logging.info(f"Processed offer_id={ad.offer_id}")

        time.sleep(1 + random.random())  # polite delay to avoid overwhelming the server
