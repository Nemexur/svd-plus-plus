from typing import Any, Optional

import haiku as hk
import jax.numpy as jnp

from svd_plus_plus.model.typing import Batch


class SvdModel(hk.Module):
    def __init__(
        self, embedders: dict[str, hk.Embed], stats: dict[str, Any], name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self._embedders = embedders
        self._stats = stats
        self._bias_init = hk.initializers.Constant(
            (self._stats["max_rating"] - self._stats["min_rating"]) / 2
        )


class PaterekSvd(SvdModel):
    def __call__(self, batch: Batch) -> dict[str, jnp.ndarray]:
        # user, item ~ (batch size)
        # similar_explicit, similar_implicit, similar_explicit_ratings ~ (batch size, padding size)
        # user_bias, item_bias, bias ~ (batch size)
        user_bias = hk.get_parameter(
            "user_bias", shape=[self._stats["num_users"]], init=self._bias_init
        )[batch["user"]]
        item_bias = hk.get_parameter(
            "item_bias", shape=[self._stats["num_items"]], init=self._bias_init
        )[batch["item"]]
        bias = self._stats["avg_rating"] + user_bias + item_bias
        # item_q ~ (batch size, hidden size)
        item_features = self._embedders["item_q"](vocab_size=self._stats["num_items"])(
            batch["item"]
        )
        # mask_item_x ~ (batch size, num similar items)
        mask_item_x = (batch["similar_explicit"] > 0).astype(jnp.float32)
        # item_x ~ (batch size, num similar items, hidden size)
        item_x = self._embedders["item_x"](vocab_size=self._stats["num_items"])(
            batch["similar_explicit"]
        )
        # user_features ~ (batch size, hidden size)
        user_features = jnp.einsum(
            "bnh,bn,b->bh",
            item_x,
            mask_item_x,
            1 / (jnp.sqrt(mask_item_x.sum(axis=-1)) + 1e-13),
        )
        output = bias + jnp.einsum("bh,bh->b", user_features, item_features)
        return {
            "output": output,
            "user_bias": user_bias,
            "item_bias": item_bias,
        }


class AsymmetricSvd(SvdModel):
    def __call__(self, batch: Batch) -> dict[str, jnp.ndarray]:
        # user, item ~ (batch size)
        # similar_explicit, similar_implicit, similar_explicit_ratings ~ (batch size, padding size)
        # user_bias, item_bias ~ (batch size)
        user_bias = hk.get_parameter(
            "user_bias", shape=[self._stats["num_users"]], init=self._bias_init
        )
        # item_bias ~ (batch size)
        item_bias = hk.get_parameter(
            "item_bias", shape=[self._stats["num_items"]], init=self._bias_init
        )
        # Bias for target users and items
        # bias ~ (batch size)
        bias = self._stats["avg_rating"] + user_bias[batch["user"]] + item_bias[batch["item"]]
        # mask_item_x, mask_item_y ~ (batch size, num similar items)
        mask_item_x = (batch["similar_explicit"] > 0).astype(jnp.float32)
        mask_item_y = (batch["similar_implicit"] > 0).astype(jnp.float32)
        # Bias for users and similar items
        # similar_explicit_bias ~ (batch size, num similar items)
        similar_explicit_bias = (
            self._stats["avg_rating"]
            + jnp.expand_dims(user_bias[batch["user"]], axis=-1)
            + item_bias[batch["similar_explicit"]]
        )
        # item_features ~ (batch size, hidden size)
        item_features = self._embedders["item_q"](vocab_size=self._stats["num_items"])(
            batch["item"]
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
        # user_features ~ (batch size, hidden size)
        user_features = similar_explicit_features + similar_implicit_features
        output = bias + jnp.einsum("bh,bh->b", user_features, item_features)
        return {
            "output": output,
            "user_bias": user_bias[batch["user"]],
            "item_bias": item_bias[batch["item"]],
        }


class SvdPlusPlus(SvdModel):
    def __call__(self, batch: Batch) -> dict[str, jnp.ndarray]:
        # user, item ~ (batch size)
        # similar_explicit, similar_implicit, similar_explicit_ratings ~ (batch size, padding size)
        # user_bias, item_bias ~ (batch size)
        user_bias = hk.get_parameter(
            "user_bias", shape=[self._stats["num_users"]], init=self._bias_init
        )[batch["user"]]
        item_bias = hk.get_parameter(
            "item_bias", shape=[self._stats["num_items"]], init=self._bias_init
        )[batch["item"]]
        # Bias for target users and items
        # bias ~ (batch size)
        bias = self._stats["avg_rating"] + user_bias + item_bias
        # mask_item_y ~ (batch size, num similar items)
        mask_item_y = (batch["similar_implicit"] > 0).astype(jnp.float32)
        # item_features ~ (batch size, hidden size)
        item_features = self._embedders["item_q"](vocab_size=self._stats["num_items"])(
            batch["item"]
        )
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
        output = bias + jnp.einsum("bh,bh->b", user_features, item_features)
        return {
            "output": output,
            "user_bias": user_bias,
            "item_bias": item_bias,
        }
