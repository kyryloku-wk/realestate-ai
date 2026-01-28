import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from tqdm.notebook import tqdm

# SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?distanceRadius=75&limit=72&ownerTypeSingleSelect=ALL&areaMin=50&by=LATEST&direction=DESC"  # можно добавить параметры
# SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?distanceRadius=75&by=LATEST&direction=DESC"
# SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?limit=72&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"

SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?limit=72&ownerTypeSingleSelect=ALL&areaMin=55&priceMax=950000&roomsNumber=%5BTHREE%2CFOUR%2CFIVE%2CSIX_OR_MORE%5D&by=LATEST&direction=DESC"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}

geolocator = Nominatim(user_agent="otodom_scraper")


def get_listing_urls(page: int = 1) -> list[str]:
    """Получаем ссылки на объявления с заданной страницы."""
    url = f"{SEARCH_URL}?page={page}"
    r = requests.get(url, headers=HEADERS)

    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a[data-cy='listing-item-link']"):
        href = a.get("href")
        if href and href.startswith("/pl/oferta"):
            links.append("https://www.otodom.pl" + href)
    return links


def get_listing_urls(page: int = 1):
    """Получаем ссылки на объявления с заданной страницы."""
    url = f"{SEARCH_URL}%3Fpage%3D2&page={page}"
    r = requests.get(url, headers=HEADERS)
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

            offers.append({"id": id_like, "url": full_url})

    return offers


def parse_listing(url: str):
    """Парсим одно объявление и возвращаем dict с данными."""
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    data = {}

    # ✅ Цена
    price_tag = soup.find("strong", {"data-cy": "adPageHeaderPrice"})
    if price_tag:
        data["price"] = int(re.sub(r"\D", "", price_tag.text))

    # ✅ Цена за м²
    price_m2_tag = soup.find("div", {"aria-label": "Cena za metr kwadratowy"})
    if price_m2_tag:
        data["price_per_m2"] = int(re.sub(r"\D", "", price_m2_tag.text))

    # ✅ Адрес (хлебные крошки)
    breadcrumbs = soup.find_all("a", {"data-cy": "breadcrumb"})
    if breadcrumbs:
        if len(breadcrumbs) >= 3:
            data["city"] = breadcrumbs[-2].text.strip()
            data["district"] = breadcrumbs[-1].text.strip()

    # ✅ Полное описание (Opis)
    script = soup.find("script", id="__NEXT_DATA__")
    json_data = json.loads(script.string)
    ad = json_data["props"]["pageProps"]["ad"]
    data["description"] = ad.get("description")

    # ✅ Сбор всех характеристик (они идут в блоках с label + value)
    for container in soup.find_all("div", class_=re.compile("css-1xw0jqp")):
        labels = container.find_all("p", class_=re.compile("css-1airkmu"))
        if len(labels) >= 2:
            key = labels[0].text.strip(": ")
            val = labels[1].text.strip()
            data[key] = val

    # ✅ Приводим ключи к нормализованным названиям
    normalize_map = {
        "Powierzchnia": "area",
        "Liczba pokoi": "rooms",
        "Ogrzewanie": "heating",
        "Piętro": "floor",
        "Czynsz": "czynsz",
        "Stan wykończenia": "finish_status",
        "Rynek": "market_type",
        "Forma własności": "ownership",
        "Dostępne od": "available_from",
        "Typ ogłoszeniodawcy": "seller_type",
        "Informacje dodatkowe": "extra_info",
        "Rok budowy": "year_built",
        "Winda": "lift",
        "Rodzaj zabudowy": "building_type",
        "Materiał budynku": "building_material",
        "Okna": "windows",
        "Certyfikat energetyczny": "energy_certificate",
        "Bezpieczeństwo": "security",
        "Wyposażenie": "equipment",
        "Zabezpieczenia": "security_devices",
        "Media": "media",
    }

    renamed = {}
    for k, v in list(data.items()):
        if k in normalize_map:
            renamed[normalize_map[k]] = v
            del data[k]
    data.update(renamed)

    # ✅ Разбиваем списки (для extra_info, equipment и т.д.)
    for field in ["extra_info", "equipment", "media", "security", "security_devices"]:
        if field in data:
            data[field] = [x.strip() for x in data[field].split(",") if x.strip()]

    # ✅ Преобразуем площадь и цену к числам
    if "area" in data:
        data["area"] = float(re.sub(r"[^\d.,]", "", data["area"]).replace(",", "."))

    if "czynsz" in data:
        data["czynsz"] = int(re.sub(r"\D", "", data["czynsz"]))

    if "year_built" in data:
        data["year_built"] = int(re.sub(r"\D", "", data["year_built"]))

    # ✅ Геокодирование в latitude и longitude
    if "city" in data:
        address = f"{data.get('district', '')}, {data['city']}, Poland"
        try:
            location = geolocator.geocode(address)
            if location:
                data["latitude"] = location.latitude
                data["longitude"] = location.longitude
        except:
            pass

    return data


# parse_listing("https://www.otodom.pl/pl/oferta/dwa-poziomu-taras-15-metrow-gotowe-bez-pcc-ID4xiwX")


def scrape_otodom(all_data, pages: int = 1):
    parsed_ids = set()
    for page in tqdm(range(1, pages + 1)):
        print(f"Парсим страницу {page}...")
        try:
            links = get_listing_urls(page)
            for elem in links:
                id = elem["id"]
                link = elem["url"]

                if id in parsed_ids:
                    continue
                else:
                    parsed_ids.add(id)
                try:
                    time.sleep(1)
                    d = parse_listing(link)
                    d["id"] = id
                    d["url"] = link
                    all_data.append(d)
                except Exception as e:
                    print(f"Ошибка при обработке ссылки {link}: {e}")
        except Exception as e:
            print(f"Ошибка при получении ссылок для страницы {page}: {e}")

    return df


all_data = []
res = scrape_otodom(all_data, pages=27)

df = pd.DataFrame(all_data)
df.to_csv("2025_13_08_otodom_krakow_myfilters.csv", index=False)
print("✅ Данные сохранены в otodom_listings.csv")
df.head()
