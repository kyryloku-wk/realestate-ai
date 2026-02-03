import json
import re

from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="otodom_scraper")


def parse_html(html_text: str):
    """Парсим одно объявление и возвращаем dict с данными."""
    soup = BeautifulSoup(html_text, "html.parser")

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


if __name__ == "__main__":
    from minio import get_bucket_name, load_html
    from postgres import query_to_dataframe

    df = query_to_dataframe("SELECT * FROM html_files ORDER BY id DESC LIMIT 1").iloc[0]
    print(df["url"])
    file_string = load_html(get_bucket_name(), df["minio_key"]).decode("utf-8", errors="replace")

    result = parse_html(file_string)

    print(result.keys())
