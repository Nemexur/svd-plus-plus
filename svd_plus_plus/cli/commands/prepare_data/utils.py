from collections import defaultdict
import json
from pathlib import Path

from loguru import logger

from svd_plus_plus.datasets.utils import DatasetParts, SimilarItems


def do_prepare_data(directory: Path, use_exp_decay: bool, top_k: int, save_dir: Path) -> None:
    logger.info("Load dataset parts.")
    parts = DatasetParts.from_directory(directory, save_dir=save_dir)
    logger.info("Get similar items.")
    similar_items = SimilarItems(parts.train, top_k, use_exp_decay=use_exp_decay).build()
    logger.info("Build stats.")
    stats = {}
    stats["min_rating"] = float(parts.train.df.rating.min())
    stats["max_rating"] = float(parts.train.df.rating.max())
    stats["num_users"] = int(parts.train.explicit_sparse.shape[0])
    stats["num_items"] = int(parts.train.explicit_sparse.shape[-1])
    stats["avg_rating"] = float(parts.train.df.rating.mean())
    stats["explicit_similar"] = similar_items.explicit_similar_items
    stats["implicit_similar"] = similar_items.implicit_similar_items
    stats["user_items"] = defaultdict(dict)
    for row in parts.train.df.itertuples():
        stats["user_items"][str(row.user)][str(row.item)] = float(row.rating)
    with (save_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)
