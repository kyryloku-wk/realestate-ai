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
            ("images_small", self.simple_get_field),
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


if __name__ == "__main__":
    df = query_to_dataframe("""SELECT * FROM listings_bronze""")
    final_df = process_payloads(df["payload"])
    save_df_to_postgres(final_df, table="extracted_payload", if_exists="replace", chunksize=100)
