import json
import logging
from dataclasses import asdict
from datetime import datetime

import dotenv
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from realestateai.data.bronze_to_silver import apply_category_features
from realestateai.data.create_training_dataset import DatasetProcessor, s3_storage_options_auto
from realestateai.feature_engineering.list_of_strings_encoder import ListOfStringsMultiHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
dotenv.load_dotenv()
set_config(transform_output="pandas")


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


def get_features_by_types():
    return {
        "target": "price_per_m",
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


def calculate_metrics(preds: np.ndarray, y: pd.Series) -> dict:
    preds = np.asarray(preds)
    y_true = np.asarray(y)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    return {
        "mae": float(mean_absolute_error(y_true, preds)),
        "mape": float(mean_absolute_percentage_error(y_true, preds)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, preds)),
        "median_ae": float(median_absolute_error(y_true, preds)),
    }


def safe_jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


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
        target = features_by_types["target"]

        # ---------- IMPORTANT: avoid leakage ----------
        y = df[target].copy()
        X = df.drop(columns=[target, "price"]).reset_index(drop=True).copy()

        # ---------- Log dataset/meta ----------
        mlflow.log_param("target", target)
        mlflow.log_param("rows", int(len(df)))
        mlflow.log_param("cols_total", int(df.shape[1]))
        mlflow.log_param("cv_folds", 15)

        # Логируем инфо о датасете/версии (как есть, без предположений о структуре)
        dataset_info_dict = asdict(info)  # info: DatasetVersionInfo
        mlflow.log_dict(dataset_info_dict, "dataset/version_info.json")

        mlflow.log_dict(features_by_types, "features_by_types.json")

        # ---------- Build pipeline ----------
        data_preprocessor = get_data_preprocessor(features_by_types)

        lgbm_params = dict(
            importance_type="gain",
            random_state=42,
            n_jobs=15,
            min_child_samples=10,
            verbosity=1,
            n_estimators=200,
        )
        mlflow.log_params(lgbm_params)

        model = Pipeline(
            [
                ("preprocessor", data_preprocessor),
                ("model", lgb.LGBMRegressor(**lgbm_params)),
            ]
        )

        # ---------- CV ----------
        cv = KFold(n_splits=15, shuffle=True, random_state=42)
        y_preds_oof = cross_val_predict(model, X, y, cv=cv, n_jobs=15)

        cv_metrics = calculate_metrics(y_preds_oof, y)
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
        logging.info(f"CV Metrics: {cv_metrics}")

        # Можно залогировать OOF предикты как артефакт (удобно для анализа ошибок)
        oof_df = pd.DataFrame({"y_true": y.values, "y_pred_oof": y_preds_oof})
        oof_path = "garbage/oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        mlflow.log_artifact(oof_path)

        mlflow.lightgbm.autolog()

        # ---------- Fit final ----------
        model.fit(X, y)

        # ---------- Feature importance ----------
        pre = model.named_steps["preprocessor"]
        lgbm = model.named_steps["model"]

        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = None

        if feature_names is not None and hasattr(lgbm, "feature_importances_"):
            fi = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_gain": lgbm.feature_importances_.astype(float),
                }
            ).sort_values("importance_gain", ascending=False)
            fi_path = "feature_importance_gain.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

        # ---------- Log final train metrics (на трейне, просто как reference) ----------
        y_pred_train = model.predict(X)
        train_metrics = calculate_metrics(y_pred_train, y)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        # ---------- Log model ----------
        # input_example: маленький сэмпл “сырых” данных до препроцессинга
        input_example = X.head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=None,  # если захочешь в Model Registry — можно указать имя
        )

        logging.info("Training finished and logged to MLflow.")
