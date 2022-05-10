import distrax as D
from einops import reduce
import haiku as hk
import jax
import jax.numpy as jnp
import optax


@jax.jit
def mse_loss(
    output: jnp.ndarray,
    target: jnp.ndarray,
    state: hk.State,
) -> jnp.ndarray:
    weights = state["weights"]
    # output, target, weights ~ (batch size)
    return reduce(weights * optax.l2_loss(output, target), "batch ->", reduction="mean")


# FIXME: Does not work with current DataPipe.
@jax.jit
def warp_loss(
    output: jnp.ndarray,
    target: jnp.ndarray,
    state: hk.State,
    max_num_trials: int = None,
    alpha: float = 1.0,
) -> jnp.ndarray:
    pos_items_idx = state["pos_items_idx"]
    neg_items_idx = state["neg_items_idx"]
    num_items = target.shape[0]
    rank_weights = jnp.log(jnp.cumsum(1.0 / jnp.arange(1, num_items + 1), axis=-1))
    # Set max number of trials and samplint distribution
    max_num_trials = max_num_trials or num_items - 1
    weights = jnp.full(num_items, fill_value=1.0 / (num_items - target.sum()))
    weights = weights.at[pos_items_idx].set(0.0)
    # Transform rank into loss with weightening function (L)
    # L ~ (num items)
    L = jnp.zeros(num_items)
    # samples_scores ~ (max num trials)
    samples_scores = output[
        D.Categorical(probs=weights).sample(seed=state["rng_key"], sample_shape=max_num_trials)
    ]
    sample_score_margin = (
        (alpha + samples_scores - jnp.expand_dims(output[pos_items_idx], axis=-1)) > 0
    ).astype(jnp.int32)
    # Pick first nonzero
    # rejection ~ (num positives)
    rejection = (sample_score_margin != 0).argmax(axis=-1)
    # Select rank for sample (+ 1 because we start from 0)
    # target_rank_loss ~ (num positives)
    target_rank_loss = rank_weights[(num_items - 1) // (rejection + 1)]
    # Check for zero in sample_score_margin
    # if we didn't manage to sample negative with greater rank
    maybe_zero = sample_score_margin[jnp.arange(pos_items_idx.shape[0]), rejection]
    target_rank_loss = jnp.where(maybe_zero == 0, 0.0, target_rank_loss)
    # Update L
    L = L.at[pos_items_idx].set(target_rank_loss)
    # Compute loss
    return jnp.einsum(
        "p,pn->",
        L[pos_items_idx],
        jax.nn.relu(
            alpha + output[neg_items_idx] - jnp.expand_dims(output[pos_items_idx], axis=-1)
        ),
    )
