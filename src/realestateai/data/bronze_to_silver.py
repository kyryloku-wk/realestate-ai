from collections.abc import Mapping
from typing import Any

import pandas as pd

from realestateai.data.postgres.listings_silver import save_df_to_postgres
from realestateai.data.postgres.utils import query_to_dataframe


class ExtractPayload:
    def simple_get_field(self, output_dict, field_name, data_dict):
        output_dict[field_name] = data_dict.get(field_name, None)

    def extract_char_fields(self, output_dict, field_name, data_dict):
        obj = data_dict.get(field_name)
        for k, v in obj.items():
            output_dict[k] = v.get("value")

    def extract_values_fields(self, output_dict, field_name, data_dict):
        obj = data_dict.get(field_name)
        for k, v in obj.items():
            values = v.get("values")
            for elem in values:
                if "::" in elem:
                    key, value = elem.split("::", 1)
                    output_dict[key] = value
                else:
                    output_dict[k] = elem

    def extract_property_raw(
        self, output_dict: dict[str, Any], field_name: str, data_dict: Mapping[str, Any]
    ) -> None:
        """
        Хотим:
        - property_raw: расплющить весь объект data_dict["property_raw"]
        - buildingProperties: расплющить либо data_dict["buildingProperties"],
            либо data_dict["property_raw"]["buildingProperties"] (в твоём примере оно там)

        Ключи делаем плоскими, например:
        property_raw__condition = "TO_RENOVATION"
        property_raw__area__value = 56.25
        property_raw__buildingProperties__year = 1970
        buildingProperties__year = 1970
        buildingProperties__security__ = "ANTI_BURGLARY_DOOR"
        """

        def flatten(obj: Any, prefix: str) -> None:
            # словарь
            if isinstance(obj, Mapping):
                for kk, vv in obj.items():
                    if kk == "__typename":
                        # почти всегда мусор для ML
                        continue
                    flatten(vv, prefix + str(kk) + "__")
                return

            # список/кортеж
            if isinstance(obj, list) or isinstance(obj, tuple):
                output_dict[prefix] = obj
                return

            # скаляр
            if prefix.endswith("__"):
                prefix_key = prefix[:-2]
            else:
                prefix_key = prefix
            output_dict[prefix_key] = obj

        if field_name == "property_raw":
            obj = data_dict.get("property_raw")
            if obj is None:
                output_dict["property_raw"] = None
                return
            flatten(obj, "property_raw__")
            return

        if field_name == "buildingProperties":
            obj = data_dict.get("buildingProperties")
            if obj is None:
                pr = data_dict.get("property_raw")
                if isinstance(pr, Mapping):
                    obj = pr.get("buildingProperties")

            if obj is None:
                output_dict["buildingProperties"] = None
                return

            flatten(obj, "buildingProperties__")
            return

        # fallback (если позже добавишь что-то ещё в этот extractor)
        obj = data_dict.get(field_name)
        if obj is None:
            output_dict[field_name] = None
            return
        flatten(obj, field_name + "__")

    def proccess(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        fields_processing = [
            ("char", self.extract_char_fields),
            ("top_info", self.extract_values_fields),
            ("additional_info", self.extract_values_fields),
            ("ad_id", self.simple_get_field),
            ("url", self.simple_get_field),
            ("status", self.simple_get_field),
            ("created_at", self.simple_get_field),
            ("modified_at", self.simple_get_field),
            ("pushed_up_at", self.simple_get_field),
            ("title", self.simple_get_field),
            ("seo_title", self.simple_get_field),
            ("seo_description", self.simple_get_field),
            ("description_text", self.simple_get_field),
            ("market", self.simple_get_field),
            ("advertiser_type", self.simple_get_field),
            ("advert_type", self.simple_get_field),
            ("exclusive_offer", self.simple_get_field),
            ("latitude", self.simple_get_field),
            ("longitude", self.simple_get_field),
            ("street", self.simple_get_field),
            ("street_number", self.simple_get_field),
            ("district", self.simple_get_field),
            ("city", self.simple_get_field),
            ("county", self.simple_get_field),
            ("province", self.simple_get_field),
            ("postal_code", self.simple_get_field),
            ("location_text", self.simple_get_field),
            ("features", self.simple_get_field),
            ("agency_name", self.simple_get_field),
            ("price_pln", self.simple_get_field),
            ("area_m2", self.simple_get_field),
            ("price_per_m2_pln", self.simple_get_field),
            ("rooms", self.simple_get_field),
            ("building_floors", self.simple_get_field),
            ("year_built", self.simple_get_field),
            ("rent_pln", self.simple_get_field),
            ("floor", self.simple_get_field),
            ("lift", self.simple_get_field),
            ("property_raw", self.extract_property_raw),
            ("buildingProperties", self.extract_property_raw),
        ]
        out: dict[str, Any] = {}
        for field_name, fn in fields_processing:
            fn(out, field_name, payload)
        return out


def process_payloads(payload_column: pd.Series) -> pd.DataFrame:
    res = pd.DataFrame(payload_column.apply(lambda x: ExtractPayload().proccess(x)).to_list())
    return res


def transform_payloads_df(payloads_df: pd.DataFrame) -> pd.DataFrame:
    new_df = payloads_df.copy()
    # technical features
    new_df["ad_id"] = new_df["ad_id"].astype("Int64")
    new_df["url"] = new_df["url"].astype("string")
    new_df["status"] = new_df["status"].str.lower().astype("category")  # actuve or inactive
    new_df["created_at"] = pd.to_datetime(new_df["created_at"], errors="coerce")
    new_df["modified_at"] = pd.to_datetime(new_df["modified_at"], errors="coerce")
    new_df["pushed_up_at"] = pd.to_datetime(new_df["pushed_up_at"], errors="coerce")

    # appartments feature
    new_df["m"] = new_df["m"].astype("Float32").round(2)
    new_df["price"] = pd.to_numeric(new_df["price"], errors="coerce").round(0).astype("Int32")
    new_df["floor"] = pd.to_numeric(new_df["floor"], errors="coerce").round(0).astype("Int32")
    new_df["rooms_num"] = new_df["rooms_num"].astype("Int32")
    new_df["price_per_m"] = (
        pd.to_numeric(new_df["price_per_m"], errors="coerce").round(2).astype("Float32")
    )

    new_df["property_type"] = (
        new_df["property_raw__type"].str.lower().astype("category")
    )  # ALL -> flat
    new_df["property_condition"] = (
        new_df["property_raw__condition"].str.lower().astype("category")
    )  # READY_TO_USE     376 TO_COMPLETION    331 TO_RENOVATION     52
    new_df["property_ownership"] = (
        new_df["property_raw__ownership"].str.lower().astype("category")
    )  # FULL_OWNERSHIP       673 LIMITED_OWNERSHIP     32 USUFRUCT               1
    new_df["property_areas"] = new_df[
        "property_raw__properties__areas__"
    ]  # list type -> [balcony, usable_room, garage, garden, terrace]
    new_df["property_kitchen"] = (
        new_df["property_raw__properties__kitchen"].str.lower().astype("category")
    )  # separate 255
    new_df["property_equipment"] = new_df[
        "property_raw__properties__equipment__"
    ]  # list type -> [furniture, washing_machine, dishwasher, fridge, stove, oven]

    # location features
    new_df["latitude"] = (
        pd.to_numeric(new_df["latitude"], errors="coerce").round(6).astype("Float32")
    )  # no nulls for latitude
    new_df["longitude"] = (
        pd.to_numeric(new_df["longitude"], errors="coerce").round(6).astype("Float32")
    )  # same for longitude
    new_df["street"] = new_df["street"].astype("string")  # ul. Stefana Banacha
    new_df["district"] = new_df["district"].astype("string")  # Prądnik Biały
    new_df["city"] = new_df["city"].astype("string")  # Kraków
    new_df["county"] = new_df["county"].astype("string")  # Kraków
    new_df["province"] = new_df["province"].astype("string")  # małopolskie

    # text features
    new_df["title"] = new_df["title"].astype("string")
    new_df["seo_description"] = new_df["seo_description"].astype("string")
    new_df["description_text"] = new_df["description_text"].astype("string")

    # building properties

    new_df["windows_type"] = new_df["windows_type"].str.lower().astype("category")
    new_df["building_type"] = new_df["building_type"].str.lower().astype("category")
    new_df["build_year"] = new_df["build_year"].astype("Int32")
    new_df["building_floors_num"] = new_df["building_floors_num"].astype("Int32")
    new_df["lift"] = new_df["lift"].astype("boolean")  # True     515 False    352
    new_df["building_heating"] = (
        new_df["property_raw__buildingProperties__heating"].str.lower().astype("category")
    )  # URBAN    673 ELECTRIC  28 GAS      12 OTHER    11
    new_df["building_material"] = (
        new_df["property_raw__buildingProperties__material"].str.lower().astype("category")
    )
    # BREEZEBLOCK 121BRICK 97CONCRETE_PLATE 58OTHER 42REINFORCED_CONCRETE 20CONCRETE 20CELLULAR_CONCRETE 14SILIKAT 7
    new_df["energy_certificate"] = (
        new_df["energy_certificate"].str.lower().astype("category")
    )  # exempt    62a         21aplus     18b          2c          1

    # org property features
    new_df["free_from"] = pd.to_datetime(new_df["free_from"], errors="coerce")
    new_df["market"] = (
        new_df["market"].str.lower().astype("category")
    )  # secondary    510 primary      357
    new_df["construction_status"] = (
        new_df["construction_status"].str.lower().astype("category")
    )  # ready_to_use 376 to_completion 331 to_renovation 52
    new_df["advertiser_type"] = (
        new_df["advertiser_type"].str.lower().astype("category")
    )  # business    785 private      82
    new_df["advert_type"] = (
        new_df["advert_type"].str.lower().astype("category")
    )  # AGENCY 753 PRIVATE 93 DEVELOPER_UNIT 21
    new_df["agency_name"] = new_df["agency_name"].astype(
        "string"
    )  # Semaco Real Estate 44, 153 unique values

    new_df["rent"] = (
        pd.to_numeric(new_df["rent_pln"], errors="coerce").round(2).astype("Float32")
    )  # 60% available -> 600, 800, ...

    # another features
    new_df["security_types"] = new_df[
        "property_raw__buildingProperties__security__"
    ]  # list type -> [ANTI_BURGLARY_DOOR, ENTRYPHONE, MONITORING]
    new_df["features"] = new_df["features"]  # list type -> several string elements
    new_df["building_conveniences"] = new_df[
        "property_raw__buildingProperties__conveniences__"
    ]  # list type -> [LIFT, INTERNET] -> remove probably
    new_df["internet"] = (
        new_df["building_conveniences"]
        .apply(lambda x: "INTERNET" in x if isinstance(x, list) else False)
        .astype("boolean")
    )  # True  122 False 775
    new_df["garage"] = new_df["extras_types-85"].isna()  # bool True     251
    new_df["extra_feature"] = new_df["extras_types"].str.lower().astype("category")
    # balcony 181separate_kitchen    170 144air_conditioning    130basement  83terrace 36garden 35two_storey 23
    new_df["media_types"] = (
        new_df["media_types"].str.lower().astype("category")
    )  # cable-television    188internet 177phone 129
    new_df["equipment_types"] = new_df["equipment_types"].str.lower().astype("category")
    # oven               106furniture 59tv 58stove 48dishwasher 35washing_machine     13
    new_df["remote_services"] = new_df["remote_services"].astype(
        "category"
    )  # ?? there only "1"    201 times

    new_df = new_df.drop(
        columns=[
            "floor_no",
            "area",
            "building_type-15",
            "seo_title",
            "exclusive_offer",
            "location_text",
            "price_pln",
            "area_m2",
            "price_per_m2_pln",
            "rooms",
            "building_floors",
            "year_built",
            "property_raw__area__unit",
            "property_raw__area__value",
            "property_raw__type",
            "property_raw__costs__",
            "property_raw__condition",
            "property_raw__ownership",
            "property_raw__properties__areas__",
            "property_raw__properties__floor",
            "property_raw__properties__rooms__",
            "property_raw__properties__kitchen",
            "property_raw__properties__parking__",
            "property_raw__properties__equipment__",
            "property_raw__properties__numberOfRooms",
            "property_raw__properties__windowsOrientation__",
            "property_raw__buildingProperties__type",
            "property_raw__buildingProperties__year",
            "property_raw__buildingProperties__heating",
            "property_raw__buildingProperties__windows__",
            "property_raw__buildingProperties__material",
            "property_raw__buildingProperties__security__",
            "property_raw__buildingProperties__conveniences__",
            "property_raw__buildingProperties__numberOfFloors",
            "rent_pln",
            "building_ownership",
            "extras_types-85",
            "extras_types",
            "construction_status-67",
            "property_raw__rent__value",
            "property_raw__rent__currency",
            "building_material-69",
            "property_raw__rent",
            "property_raw__properties__type",
            "postal_code",
            "property_raw__id",
            "flat_projection",
            "street_number",
            "flat_number",
            "buildingProperties__type",
            "buildingProperties__year",
            "buildingProperties__heating",
            "buildingProperties__windows__",
            "buildingProperties__material",
            "buildingProperties__security__",
            "buildingProperties__conveniences__",
            "buildingProperties__numberOfFloors",
        ]
    )

    return new_df


def apply_category_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_features = [
        "market",
        "windows_type",
        "building_type",
        "construction_status",
        "advertiser_type",
        "status",
        "advert_type",
        "building_material",
        "media_types",
        "equipment_types",
        "remote_services",
        "energy_certificate",
        "property_type",
        "property_condition",
        "property_ownership",
        "property_kitchen",
        "building_heating",
        "extra_feature",
    ]
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


if __name__ == "__main__":
    df = query_to_dataframe("""SELECT * FROM listings_bronze""")
    final_df = process_payloads(df["payload"])
    final_df = transform_payloads_df(final_df)
    save_df_to_postgres(final_df, table="listings_silver", if_exists="replace")

    loaded_df = query_to_dataframe("SELECT * FROM listings_silver")
    loaded_df = apply_category_features(loaded_df)
    print(loaded_df)
    # print(final_df.dtypes)
