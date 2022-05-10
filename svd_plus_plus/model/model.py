from typing import Any, Optional
from abc import ABC, abstractmethod

import haiku as hk
import jax.numpy as jnp

from svd_plus_plus.model.typing import Batch


class SvdModel(ABC, hk.Module):
    def __init__(
        self, embedders: dict[str, hk.Embed], stats: dict[str, Any], name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self._embedders = embedders
        self._stats = stats
        self._bias_init = hk.initializers.Constant(
            (self._stats["max_rating"] - self._stats["min_rating"]) / 2
        )

    @property
    def user_bias(self) -> jnp.ndarray:
        return hk.get_parameter("user_bias", shape=[self._stats["num_users"]], init=self._bias_init)

    @property
    def item_bias(self) -> jnp.ndarray:
        return hk.get_parameter("item_bias", shape=[self._stats["num_items"]], init=self._bias_init)

    def get_bias(self, batch: Batch) -> jnp.ndarray:
        return (
            self._stats["avg_rating"]
            + self.user_bias[batch["user"]]
            + self.item_bias[batch["item"]]
        )

    @abstractmethod
    def get_user_features(self, batch: Batch) -> jnp.ndarray:
        pass

    @abstractmethod
    def get_item_features(self, batch: Batch) -> jnp.ndarray:
        pass

    def __call__(self, batch: Batch) -> dict[str, jnp.ndarray]:
        bias = self.get_bias(batch)
        user_features, item_features = self.get_user_features(batch), self.get_item_features(batch)
        return {
            "output": bias + jnp.einsum("bh,bh->b", user_features, item_features),
            "user_bias": self.user_bias[batch["user"]],
            "item_bias": self.item_bias[batch["item"]],
        }


class PaterekSvd(SvdModel):
    @hk.transparent
    def get_user_features(self, batch: Batch) -> jnp.ndarray:
        # mask_item_x ~ (batch size, num similar items)
        mask_item_x = (batch["similar_explicit"] > 0).astype(jnp.float32)
        # item_x ~ (batch size, num similar items, hidden size)
        item_x = self._embedders["item_x"](vocab_size=self._stats["num_items"])(
            batch["similar_explicit"]
        )
        # output ~ (batch size, hidden size)
        return jnp.einsum(
            "bnh,bn,b->bh",
            item_x,
            mask_item_x,
            1 / (jnp.sqrt(mask_item_x.sum(axis=-1)) + 1e-13),
        )

    @hk.transparent
    def get_item_features(self, batch: Batch) -> jnp.ndarray:
        # output ~ (batch size, hidden size)
        return self._embedders["item_q"](vocab_size=self._stats["num_items"])(batch["item"])


class AsymmetricSvd(SvdModel):
    @hk.transparent
    def get_user_features(self, batch: Batch) -> jnp.ndarray:
        # mask_item_x, mask_item_y ~ (batch size, num similar items)
        mask_item_x = (batch["similar_explicit"] > 0).astype(jnp.float32)
        mask_item_y = (batch["similar_implicit"] > 0).astype(jnp.float32)
        # Bias for users and similar items
        # similar_explicit_bias ~ (batch size, num similar items)
        similar_explicit_bias = (
            self._stats["avg_rating"]
            + jnp.expand_dims(self.user_bias[batch["user"]], axis=-1)
            + self.item_bias[batch["similar_explicit"]]
        )
        # item_x, item_y ~ (batch size, num similar items, hidden size)
        item_x = self._embedders["item_x"](vocab_size=self._stats["num_items"])(
            batch["similar_explicit"]
        )
        item_y = self._embedders["item_y"](vocab_size=self._stats["num_items"])(
            batch["similar_implicit"]
        )
        # Get representation with explicit items
        # similar_explicit_features ~ (batch size, hidden size)
        similar_explicit_features = jnp.einsum(
            "bn,bnh,bn,b->bh",
            batch["similar_explicit_ratings"] - similar_explicit_bias,
            item_x,
            mask_item_x,
            1 / (jnp.sqrt(mask_item_x.sum(axis=-1)) + 1e-13),
        )
        # Get representation with implicit items
        # similar_implicit_features ~ (batch size, hidden size)
        similar_implicit_features = jnp.einsum(
            "bnh,bn,b->bh",
            item_y,
            mask_item_y,
            1 / (jnp.sqrt(mask_item_y.sum(axis=-1)) + 1e-13),
        )
        # output ~ (batch size, hidden size)
        return similar_explicit_features + similar_implicit_features

    @hk.transparent
    def get_item_features(self, batch: Batch) -> jnp.ndarray:
        # output ~ (batch size, hidden size)
        return self._embedders["item_q"](vocab_size=self._stats["num_items"])(batch["item"])


class SvdPlusPlus(SvdModel):
    @hk.transparent
    def get_user_features(self, batch: Batch) -> jnp.ndarray:
        # mask_item_y ~ (batch size, num similar items)
        mask_item_y = (batch["similar_implicit"] > 0).astype(jnp.float32)
        # user_features ~ (batch size, hidden size)
        user_features = self._embedders["user_p"](vocab_size=self._stats["num_users"])(
            batch["user"]
        )
        # item_x, item_y ~ (batch size, num similar items, hidden size)
        item_y = self._embedders["item_y"](vocab_size=self._stats["num_items"])(
            batch["similar_implicit"]
        )
        # Get representation with implicit items
        # similar_implicit_features ~ (batch size, hidden size)
        similar_implicit_features = jnp.einsum(
            "bnh,bn,b->bh",
            item_y,
            mask_item_y,
            1 / (jnp.sqrt(mask_item_y.sum(axis=-1)) + 1e-13),
        )
        # user_features ~ (batch size, hidden size)
        user_features += similar_implicit_features
        return user_features

    @hk.transparent
    def get_item_features(self, batch: Batch) -> jnp.ndarray:
        # output ~ (batch size, hidden size)
        return self._embedders["item_q"](vocab_size=self._stats["num_items"])(batch["item"])
