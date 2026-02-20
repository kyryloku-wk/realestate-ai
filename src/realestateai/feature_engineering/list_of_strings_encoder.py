from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _is_nan(x) -> bool:
    return isinstance(x, float) and np.isnan(x)


def _parse_possible_list_string(x):
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return x
    try:
        return ast.literal_eval(s)
    except Exception:
        inner = s[1:-1].strip()
        if inner == "":
            return []
        return [p.strip() for p in inner.split(",") if p.strip()]


def _normalize_cell_to_tokens(x) -> list[str]:
    if x is None or _is_nan(x):
        return []

    x = _parse_possible_list_string(x)

    # ✅ ВАЖНО: поддержка np.ndarray / pandas containers
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        out = []
        for v in list(x):
            if v is None or _is_nan(v):
                continue
            t = str(v).strip()
            if t != "":
                out.append(t)
        return out

    if isinstance(x, str):
        t = x.strip()
        return [] if t == "" else [t]

    return [str(x).strip()]


class ListOfStringsMultiHotEncoder(BaseEstimator, TransformerMixin):
    """
    Multi-hot encoder for one or many columns containing list-of-strings (or strings that look like lists).

    Output columns are prefixed: "{feature}__{token}", plus optional "{feature}__{EMPTY/OTHER}".
    """

    def __init__(
        self,
        min_frequency: int = 1,
        add_empty: bool = True,
        add_other: bool = True,
        empty_token: str = "__EMPTY__",
        other_token: str = "__OTHER__",
        sep: str = "__",
        dtype: str = "int8",
    ):
        self.min_frequency = int(min_frequency)
        self.add_empty = bool(add_empty)
        self.add_other = bool(add_other)
        self.empty_token = empty_token
        self.other_token = other_token
        self.sep = sep
        self.dtype = dtype

    def fit(self, X, y=None):
        Xdf = self._to_df(X)
        self.input_features_ = list(Xdf.columns)

        self.vocab_ = {}  # col -> list of kept tokens
        self.kept_set_ = {}  # col -> set(kept)
        self.feature_names_out_ = []

        for col in self.input_features_:
            counts = {}
            for cell in Xdf[col].values:
                for tok in _normalize_cell_to_tokens(cell):
                    counts[tok] = counts.get(tok, 0) + 1

            kept = [t for t, c in counts.items() if c >= self.min_frequency]
            kept.sort()

            self.vocab_[col] = kept
            self.kept_set_[col] = set(kept)

            # feature names
            for t in kept:
                self.feature_names_out_.append(f"{col}{self.sep}{t}")
            if self.add_empty:
                self.feature_names_out_.append(f"{col}{self.sep}{self.empty_token}")
            if self.add_other:
                self.feature_names_out_.append(f"{col}{self.sep}{self.other_token}")

        return self

    def transform(self, X):
        self._check_is_fitted()
        Xdf = self._to_df(X)

        out = pd.DataFrame(
            data=np.zeros((len(Xdf), len(self.feature_names_out_)), dtype=self.dtype),
            columns=self.feature_names_out_,
            index=Xdf.index,
        )

        for col in self.input_features_:
            kept = self.kept_set_[col]
            empty_col = f"{col}{self.sep}{self.empty_token}"
            other_col = f"{col}{self.sep}{self.other_token}"

            for i, cell in enumerate(Xdf[col].values):
                tokens = _normalize_cell_to_tokens(cell)

                if len(tokens) == 0:
                    if self.add_empty:
                        out.iat[i, out.columns.get_loc(empty_col)] = 1
                    continue

                other_hit = False
                for tok in tokens:
                    if tok in kept:
                        fname = f"{col}{self.sep}{tok}"
                        out.iat[i, out.columns.get_loc(fname)] = 1
                    else:
                        other_hit = True

                if other_hit and self.add_other:
                    out.iat[i, out.columns.get_loc(other_col)] = 1

        return out

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        return np.array(self.feature_names_out_, dtype=object)

    # ---- helpers ----
    def _to_df(self, X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, pd.DataFrame):
            return X
        # if numpy array passed from ColumnTransformer
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            return pd.DataFrame({"x0": X})
        cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def _check_is_fitted(self):
        if not hasattr(self, "feature_names_out_"):
            raise AttributeError("ListOfStringsMultiHotEncoder is not fitted yet.")


if __name__ == "__main__":
    enc = ListOfStringsMultiHotEncoder(min_frequency=2, add_empty=True, add_other=True)

    df = pd.DataFrame(
        {
            "col": [
                ["a", "b"],
                ["a", "c"],
                [],
                None,
            ]
        }
    )
    print(df)
    print(enc.fit_transform(df["col"]))
