from typing import Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@lru_cache(maxsize=None)
def get_stats(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@dataclass
class Encoder:
    item_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_item: dict[str, str] = field(default_factory=dict)

    def add(self, item: str) -> None:
        idx = len(self.item_to_idx)
        self.item_to_idx[str(item)] = idx
        self.idx_to_item[str(idx)] = str(item)

    def encode(self, items: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(items, str):
            return self.item_to_idx[str(items)]
        return [self.item_to_idx[str(x)] for x in items]

    def decode(self, items: Union[int, list[int]]) -> Union[str, list[str]]:
        if isinstance(items, str):
            return self.idx_to_item[str(items)]
        return [self.idx_to_item[str(x)] for x in items]

    def save(self, path: Path) -> None:
        if not path.exists():
            path.mkdir()
        with (path / "item_to_idx.json").open("w", encoding="utf-8") as file:
            json.dump(self.item_to_idx, file, indent=2, ensure_ascii=False)
        with (path / "idx_to_item.json").open("w", encoding="utf-8") as file:
            json.dump(self.idx_to_item, file, indent=2, ensure_ascii=False)

    @classmethod
    def from_values(cls: "Encoder", values: np.ndarray) -> "Encoder":
        _cls = cls()
        # Add padding and oov element
        _cls.add("@PADDING@")
        _cls.add("@OOV@")
        for value in values:
            _cls.add(value)
        return _cls

    @classmethod
    def load(cls: "Encoder", path: Path) -> "Encoder":
        with (path / "item_to_idx.json").open("r", encoding="utf-8") as file:
            item_to_idx = json.load(file)
        with (path / "idx_to_item.json").open("r", encoding="utf-8") as file:
            idx_to_item = json.load(file)
        return cls(item_to_idx, idx_to_item)


@dataclass
class Dataset:
    df: pd.DataFrame
    explicit_sparse: Optional[sp.csr_matrix] = None
    binary_sparse: Optional[sp.csr_matrix] = None
    exp_decay_sparse: Optional[sp.csr_matrix] = None

    def exponential_decay(self, values: np.ndarray) -> np.ndarray:
        # Date to unix timestamp
        timestamp = (pd.to_datetime(self.df.date).astype(int) / 10**9).values
        max_val, half_life = timestamp.max(), 30 * 24 * 60 * 60
        return values * np.minimum(1.0, np.power(0.5, (max_val - timestamp) / half_life))

    def make_sparse(self, shape: Tuple[int, int]) -> "Dataset":
        explicit_values = self.df.rating.to_numpy()
        binary_values = np.ones_like(explicit_values)
        exp_decay_values = self.exponential_decay(explicit_values)
        self.explicit_sparse = sp.csr_matrix(
            (explicit_values, (self.df.user.to_numpy(), self.df.item.to_numpy())), shape=shape
        )
        self.binary_sparse = sp.csr_matrix(
            (binary_values, (self.df.user.to_numpy(), self.df.item.to_numpy())), shape=shape
        )
        self.exp_decay_sparse = sp.csr_matrix(
            (exp_decay_values, (self.df.user.to_numpy(), self.df.item.to_numpy())), shape=shape
        )
        return self

    def encode(self, encoder: Encoder, namespace: str) -> None:
        self.df[namespace] = encoder.encode(self.df[namespace].values)

    def decode(self, encoder: Encoder, namespace: str) -> None:
        self.df[namespace] = encoder.decode(self.df[namespace].values)


@dataclass
class DatasetParts:
    train: Dataset
    valid: Dataset
    test: Dataset

    # FIXME: This function does way to much
    @classmethod
    def from_directory(cls: "DatasetParts", directory: Path, save_dir: Path) -> "DatasetParts":
        # 1. Initialize Dataset parts
        parts = {
            "train": Dataset(pd.read_csv(directory / "train.csv")),
            "valid": Dataset(pd.read_csv(directory / "valid.csv")),
            "test": Dataset(pd.read_csv(directory / "test.csv")),
        }
        # 2. Build encoders if needed
        encoders_path = save_dir / "encoders"
        if not encoders_path.exists():
            encoders_path.mkdir()
        encoders = {
            "user": (
                Encoder.load(encoders_path / "user")
                if (encoders_path / "user").exists()
                else Encoder.from_values(parts["train"].df.user.unique())
            ),
            "item": (
                Encoder.load(encoders_path / "item")
                if (encoders_path / "item").exists()
                else Encoder.from_values(parts["train"].df.item.unique())
            ),
        }
        for key, encoder in encoders.items():
            encoder_path = encoders_path / key
            if not encoder_path.exists():
                encoder.save(encoder_path)
        # 3. Encode user and item
        for part in parts.values():
            for namespace in ("user", "item"):
                part.encode(encoders[namespace], namespace)
        # 4. Save datasets
        for part, dataset in parts.items():
            save_path = save_dir / f"{part}.csv"
            dataset.df.sample(frac=1).reset_index(drop=True).to_csv(save_path, index=False)
        # 5. Build sparse matrices with shape from train dataset
        shape = len(encoders["user"].item_to_idx), len(encoders["item"].item_to_idx)
        return cls(**{k: v.make_sparse(shape) for k, v in parts.items()})


class SimilarItems:
    def __init__(self, dataset: Dataset, top_k: int = 100, use_exp_decay: bool = False) -> None:
        # Compute similar items with ratings
        # Compute similar items with implicit
        self._dataset = dataset
        self._top_k = top_k
        self._use_exp_decay = use_exp_decay
        self.explicit_similar_items: dict[int, Set[int]] = None
        self.implicit_similar_items: dict[int, Set[int]] = None

    def build(self) -> "SimilarItems":
        # pearson_corr ~ (num items, num items)
        pearson_corr = self.pearson_correlation(
            self._dataset.exp_decay_sparse.tocsc()
            if self._use_exp_decay
            else self._dataset.explicit_sparse.tocsc()
        )
        np.fill_diagonal(pearson_corr, -1000)
        self.explicit_similar_items = {
            item_idx: [int(x) for x in similar]
            for item_idx, similar in enumerate(np.argsort(-pearson_corr)[:, : self._top_k])
        }
        pearson_corr = self.pearson_correlation(self._dataset.binary_sparse.tocsc())
        np.fill_diagonal(pearson_corr, -1000)
        self.implicit_similar_items = {
            item_idx: [int(x) for x in similar]
            for item_idx, similar in enumerate(np.argsort(-pearson_corr)[:, : self._top_k])
        }
        return self

    @staticmethod
    def pearson_correlation(matrix: sp.csc_matrix) -> np.ndarray:
        # Substract mean from each item rating
        num_items = matrix.shape[-1]
        interactions_per_col = np.diff(matrix.indptr)
        nonzero_cols = interactions_per_col > 0
        sum_per_col = np.asarray(matrix.sum(axis=0)).ravel()
        col_average = np.zeros_like(sum_per_col)
        col_average[nonzero_cols] = sum_per_col[nonzero_cols] / interactions_per_col[nonzero_cols]
        # Split in blocks to avoid duplicating the whole data structure
        start_col = end_col = 0
        block_size = 1000
        while end_col < num_items:
            end_col = min(num_items, end_col + block_size)
            matrix.data[matrix.indptr[start_col] : matrix.indptr[end_col]] -= np.repeat(
                col_average[start_col:end_col], interactions_per_col[start_col:end_col]
            )
            start_col += block_size
        # Dot product and divide by root of sum of squares
        item_to_item = matrix.T.dot(matrix).toarray()
        sum_of_squared = np.asarray(matrix.power(2).sum(axis=0)).ravel()
        denominator = np.sqrt(sum_of_squared * np.expand_dims(sum_of_squared, axis=-1)) + 1e-13
        return item_to_item / denominator
