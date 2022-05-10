from typing import Any, Callable, NamedTuple, Optional, Tuple
from collections import OrderedDict
from functools import partial
from pathlib import Path

from animus import ICallback, IExperiment
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from torch.utils.data import DataLoader

from svd_plus_plus.datasets.datapipe import get_stats
from svd_plus_plus.model import SvdModel
from svd_plus_plus.model.callbacks import LoggerCallback, MetricsCallback, ProgressBarCallback
from svd_plus_plus.model.metrics import Loss, RMSEMetric
from svd_plus_plus.model.typing import Batch


class BatchOutput(NamedTuple):
    loss: Optional[jnp.ndarray]
    output: dict[str, jnp.ndarray]
    state: Optional[hk.State] = None


class SvdRunner(IExperiment):
    def __init__(
        self,
        model: SvdModel,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        num_epochs: int,
        stats_path: str,
        seed: int = 13,
        callbacks: dict[str, ICallback] = None,
    ) -> None:
        super().__init__()
        self.state: hk.State = None
        self.rng_seq = hk.PRNGSequence(seed)
        self.forward = hk.transform(self._forward_fn)
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._stats = get_stats(Path(stats_path))
        # Set parameters for IExperiment
        self.callbacks = callbacks or {}
        self.seed = seed
        self.num_epochs = num_epochs
        # Extra variables
        self.batch_output: BatchOutput = None
        self.test_dataset: DataLoader = None

    def _forward_fn(self, batch: Batch) -> Tuple[Optional[jnp.ndarray], dict[str, jnp.ndarray]]:
        model = self._model(
            stats={
                k: v
                for k, v in self._stats.items()
                if k in ("num_users", "num_items", "min_rating", "avg_rating", "max_rating")
            }
        )
        output_dict = model(batch)
        if "target" not in batch:
            return None, output_dict
        target, batch_state = batch.get("target"), batch.get("state")
        batch_state["rng_key"] = hk.next_rng_key()
        loss = self._loss_fn(output_dict["output"], target, batch_state)
        return loss, output_dict

    def _setup_callbacks(self) -> None:
        self.callbacks = OrderedDict(
            {
                "metrics": MetricsCallback(metrics={"loss": Loss(), "rmse": RMSEMetric()}),
                "progress_bar": ProgressBarCallback(),
                "logger": LoggerCallback(),
                **self.callbacks,
            }
        )

    @partial(jax.jit, static_argnums=0)
    def init_state(self, rng_key: jax.random.PRNGKey, batch: Batch) -> dict[str, Any]:
        params = self.forward.init(rng_key, batch)
        opt_state = self._optimizer.init(params)
        return {"opt_state": opt_state, "params": params}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        self._setup_callbacks()
        self.state = self.init_state(next(self.rng_seq), next(iter(self.datasets["train"])))

    def on_batch_end(self, exp: "IExperiment") -> None:
        super().on_batch_end(exp)
        if self.is_train_dataset:
            self.state = self.batch_output.state

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        if self.test_dataset is not None:
            self.dataset_key, self.dataset = "test", self.test_dataset
            self._run_event("on_dataset_start")
            self.run_dataset()
            self._run_event("on_dataset_end")

    @partial(jax.jit, static_argnums=0)
    def _train_batch(
        self, state: hk.State, rng_key: jax.random.PRNGKey, batch: Batch
    ) -> BatchOutput:
        params, opt_state = state["params"], state["opt_state"]
        (loss, output), grads = jax.value_and_grad(self.forward.apply, has_aux=True)(
            params, rng_key, batch
        )
        updates, opt_state = self._optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        new_state = {"opt_state": opt_state, "params": params}
        return BatchOutput(loss, output, new_state)

    @partial(jax.jit, static_argnums=0)
    def _eval_batch(
        self, state: hk.State, rng_key: jax.random.PRNGKey, batch: Batch
    ) -> BatchOutput:
        return BatchOutput(*self.forward.apply(state["params"], rng_key, batch))

    def run_batch(self) -> None:
        self.batch_output = (
            self._train_batch(self.state, next(self.rng_seq), self.batch)
            if self.is_train_dataset
            else self._eval_batch(self.state, next(self.rng_seq), self.batch)
        )

    def run(self, datasets: dict[str, DataLoader]) -> None:
        self.test_dataset = datasets.pop("test", None)
        self.datasets = datasets
        super().run()
