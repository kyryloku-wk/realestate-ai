import logging
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

import dotenv
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from realestateai.data.bronze_to_silver import apply_category_features
from realestateai.data.create_training_dataset import DatasetProcessor, s3_storage_options_auto
from realestateai.feature_engineering.list_of_strings_encoder import ListOfStringsMultiHotEncoder
from realestateai.models.evaluation import calculate_regression_metrics
from realestateai.models.mlflow_utils import log_pd_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
dotenv.load_dotenv()
set_config(transform_output="pandas")

TARGET = "price_per_m"
CV_FOLDS = 15


def base_data_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["price_per_m"] > 500) & (df["price_per_m"] < 25000)]
    return df


def feature_preparation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["col_13"] = df["col_13"].astype(str).apply(lambda x: True if x == "y" else False)
    df["property_kitchen"] = (
        df["property_kitchen"].astype(str).apply(lambda x: True if x == "separate" else False)
    )
    return df


def get_features_by_types() -> dict[str, list[str]]:
    return {
        "float_features": ["m", "latitude", "longitude", "rent"],
        "ordinal_features": ["rooms_num", "build_year", "building_floors_num", "floor"],
        "category_features": [
            "market",
            "heating",
            "windows_type",
            "building_type",
            "building_material",
            "construction_status",
            "advertiser_type",
            "advert_type",
            "media_types",
            "equipment_types",
            "remote_services",
            "energy_certificate",
            "property_type",
            "property_condition",
            "property_ownership",
            "building_heating",
            "extra_feature",
            "district",
            "city",
            "province",
        ],
        "high_cardinality_string": ["agency_name"],
        "bool_features": ["col_13", "lift", "internet", "garage", "property_kitchen"],
        "datetime_features": ["created_at"],
        "list_of_strings_type_features": [
            "security_types",
            "features",
            "property_areas",
            "property_equipment",
            "building_conveniences",
        ],
    }


def get_input_features() -> list[str]:
    all_features = get_features_by_types()
    input_features = []
    for key in all_features:
        input_features.extend(all_features[key])
    return input_features


def get_data_preprocessor(features):
    boolean_transformer = Pipeline(
        [
            (
                "type_transformer",
                FunctionTransformer(lambda x: x.astype("boolean").astype("Float32")),
            ),
        ]
    )
    float_transformer = Pipeline(
        [("type_transformer", FunctionTransformer(lambda x: x.astype("Float32")))]
    )
    ordinal_features_transformer = Pipeline(
        [("type_transformer", FunctionTransformer(lambda x: x.astype("Int32")))]
    )
    category_features_transformer = Pipeline(
        [("type_transformer", FunctionTransformer(lambda x: x.astype("category")))]
    )
    datetime_features_transformer = Pipeline(
        [("DatetimeFeatures", DatetimeFeatures(features_to_extract=["year", "month"]))]
    )
    list_of_strings = Pipeline([("encoder", ListOfStringsMultiHotEncoder(min_frequency=10))])

    preprocessor = ColumnTransformer(
        [
            ("bools", boolean_transformer, features["bool_features"]),
            ("floats", float_transformer, features["float_features"]),
            ("ordinal", ordinal_features_transformer, features["ordinal_features"]),
            (
                "category",
                category_features_transformer,
                features["category_features"] + features["high_cardinality_string"],
            ),
            ("datetime", datetime_features_transformer, features["datetime_features"]),
            ("list_of_strings", list_of_strings, features["list_of_strings_type_features"]),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


if __name__ == "__main__":
    # ---------- MLflow setup ----------
    mlflow.set_experiment("realestate_price_per_m_lgbm")

    run_name = f"lgbm_price_per_m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Optional: autolog (может логировать лишнее, но удобно)
        # mlflow.sklearn.autolog(log_models=False)  # можно включить, если хочешь

        # ---------- Load ----------
        df, info = DatasetProcessor(storage_options=s3_storage_options_auto()).get_latest_dataset()
        df = apply_category_features(df)
        df = base_data_filter(df)
        df = feature_preparation(df)

        features_by_types = get_features_by_types()

        # ---------- IMPORTANT: avoid leakage ----------
        df = df.reset_index(drop=True)
        y = df[TARGET].copy()
        X = df[get_input_features()].copy()

        mlflow.log_param("feature_names", get_input_features())

        # - log dataset csv and parquet (for types information)
        log_pd_dataframe(X.head(5), "X_head.csv", file_format="csv")
        log_pd_dataframe(X.head(5), "X_head.parquet", file_format="parquet")

        # ---------- Log meta params ----------
        meta_params = {
            "target": TARGET,
            "n_rows": len(X),
            "cols_total": int(df.shape[1]),
            "cv_folds": CV_FOLDS,
        }
        mlflow.log_params(meta_params)

        # log dataset info
        dataset_info_dict = asdict(info)  # info: DatasetVersionInfo
        mlflow.log_dict(dataset_info_dict, "dataset/version_info.json")
        mlflow.log_dict(features_by_types, "features_by_types.json")

        # ---------- Build pipeline ----------
        data_preprocessor = get_data_preprocessor(features_by_types)

        lgbm_params = dict(
            importance_type="gain",
            random_state=42,
            n_jobs=15,
            min_child_samples=32,
            verbosity=1,
            n_estimators=200,
        )
        mlflow.log_params(lgbm_params)

        preds_df = pd.DataFrame(y)
        for quantile in [0.1, 0.5, 0.9]:
            params = deepcopy(lgbm_params)
            params["objective"] = "quantile"
            params["alpha"] = quantile

            model = Pipeline(
                [
                    ("preprocessor", data_preprocessor),
                    ("model", lgb.LGBMRegressor(**params)),
                ]
            )

            # ---------- CV ----------
            cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
            y_preds_oof = cross_val_predict(model, X, y, cv=cv, n_jobs=15)
            preds_df[f"preds_quantile_{quantile}"] = y_preds_oof

        # --- preprocess prediction df before save
        preds_df["ad_id"] = df["ad_id"]

        preds_df = preds_df.rename(columns={"preds_quantile_0.5": "predicted_price"})
        preds_df["error"] = preds_df[TARGET] - preds_df["predicted_price"]
        preds_df["interval_with"] = preds_df["preds_quantile_0.9"] - preds_df["preds_quantile_0.1"]
        preds_df["confidence_score"] = 1 - (preds_df["interval_with"] / preds_df["predicted_price"])
        preds_df["ranking_score"] = preds_df["confidence_score"] * preds_df["error"]
        preds_df["url"] = df["url"]
        preds_df = preds_df.sort_values(by="ranking_score")
        log_pd_dataframe(preds_df, "oof_predictions.csv", file_format="csv")
        log_pd_dataframe(preds_df, "oof_predictions.parquet", file_format="parquet")

        cv_metrics = calculate_regression_metrics(preds_df["predicted_price"], y)
        cv_metrics["average_confidence"] = preds_df["confidence_score"].mean()
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
        logging.info(f"CV Metrics: {cv_metrics}")

        # # ---------- Fit final ----------
        # model.fit(X, y)

        # # ---------- Log final train metrics (на трейне, просто как reference) ----------
        # y_pred_train = model.predict(X)
        # train_metrics = calculate_regression_metrics(y_pred_train, y)
        # mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        # # ---------- Log model ----------
        # # input_example: маленький сэмпл “сырых” данных до препроцессинга
        # input_example = X.head(5)

        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="model",
        #     input_example=input_example,
        #     registered_model_name=None,  # если захочешь в Model Registry — можно указать имя
        # )

        # logging.info("Training finished and logged to MLflow.")
