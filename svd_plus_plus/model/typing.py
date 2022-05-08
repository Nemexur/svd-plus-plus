from typing import Mapping

import jax.numpy as jnp

Batch = Mapping[str, jnp.ndarray]
Params = Mapping[str, jnp.ndarray]
