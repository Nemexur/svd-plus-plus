from typing import Any, Iterable
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torchdata.datapipes as dp

from svd_plus_plus.datasets.datapipe_utils import _wrap_split_argument
from svd_plus_plus.datasets.utils import get_stats


def pad_sequence(
    sequences: np.ndarray, batch_first: bool = False, padding_value: float = 0.0
) -> jnp.ndarray:
    max_len = max([s.shape[-1] for s in sequences])
    out_dims = (len(sequences), max_len)
    out_tensor = np.full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return jnp.asarray(out_tensor)


class Batch:
    """
    Class to construct batch from iterable of instances
    (each instance is a dictionary).

    Example
    -------
    Dictionary instance `{'tokens': ..., 'labels': ...}`
    then to access tokens you need to get an `attribute tokens` from `Batch` class instance.
    """

    def __init__(self, instances: Iterable[dict]) -> None:
        tensor_dict = self._as_tensor_dict_from(instances)
        self.__dict__.update(tensor_dict)

    def __repr__(self) -> str:
        cls = str(self.__class__.__name__)
        info = ", ".join(map(lambda x: f"{x[0]}={x[1]}", self.__dict__.items()))
        return f"{cls}({info})"

    @staticmethod
    def _as_tensor_dict_from(instances: Iterable[dict]) -> dict[str, list]:
        """
        Construct tensor from list of `instances` per namespace.

        Returns
        -------
        `Dict[str, List]`
            Dict in such format:
                - key: namespace id
                - value: list of torch tensors
        """
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict


class NetflixDataset:
    def __init__(
        self,
        directory: str,
        max_steps: dict[str, int],
    ) -> None:
        self.directory = Path(directory)
        self.max_steps = max_steps
        self._stats = get_stats(self.directory / "stats.json")

    def _get_similar_scored(self, user: int, item: int) -> dict[str, list[Any]]:
        user_scores = self._stats["user_items"].get(str(user))
        explicit_similar = set(int(x) for x in self._stats["explicit_similar"].get(str(item)))
        implicit_similar = set(int(x) for x in self._stats["implicit_similar"].get(str(item)))
        user_items = set(int(x) for x in user_scores)
        result = {
            "similar_explicit": list(user_items & explicit_similar),
            "similar_implicit": list(user_items & implicit_similar),
        }
        result["similar_explicit_ratings"] = [
            float(user_scores[str(x)]) for x in result["similar_explicit"]
        ]
        return result

    def _process_one_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = {
            "user": int(sample["user"]),
            "item": int(sample["item"]),
            **self._get_similar_scored(sample["user"], sample["item"]),
        }
        if "rating" in sample:
            result["target"] = float(sample["rating"])
        return result

    @_wrap_split_argument(splits=("train", "valid", "test"))
    def get_pipes(self, split: str):
        dataset_file = dp.iter.FileLister(str(self.directory), masks="*.csv").filter(
            partial(self._filter_files, split=split)
        )
        return (
            dp.iter.FileOpener(dataset_file, mode="r")
            .parse_csv_as_dict()
            .map(self._process_one_sample)
            .filter(self._filter_samples)
            .header(self.max_steps[split])
        )

    @staticmethod
    def _filter_samples(sample: dict[str, Any]) -> bool:
        return len(sample["similar_explicit"]) > 0 and len(sample["similar_implicit"]) > 0

    @staticmethod
    def _filter_files(fname: str, split: str) -> bool:
        return split in Path(fname).name


# TODO: Add support for warp loss. Probably change the way in works
# For that we need to pass a 3D tensor for similars
class NetflixDatasetCollator:
    def __init__(self, binary: bool = False) -> None:
        self._binary = binary

    def __call__(self, instances: Iterable[dict]) -> dict[str, jnp.ndarray]:
        batch = {}
        instances = Batch(instances)
        # Most important part
        batch["user"] = jnp.array(instances.user)
        batch["item"] = jnp.array(instances.item)
        batch["target"] = (
            (jnp.array(instances.target) > 0).astype(jnp.float32)
            if self._binary
            else jnp.array(instances.target)
        )
        batch["similar_explicit"] = pad_sequence(
            [np.array(x) for x in instances.similar_explicit]
        ).astype(jnp.int32)
        batch["similar_implicit"] = pad_sequence(
            [np.array(x) for x in instances.similar_implicit]
        ).astype(jnp.int32)
        batch["similar_explicit_ratings"] = pad_sequence(
            [np.array(x) for x in instances.similar_explicit_ratings]
        ).astype(jnp.float32)
        # Define state
        batch["state"] = state = {}
        state["weights"] = (batch["target"] > 0).astype(jnp.float32)
        return batch
