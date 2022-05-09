from typing import Any, Callable, NamedTuple, Optional, Tuple
from functools import partial
from pathlib import Path

from alive_progress import alive_bar
from animus import ICallback, IExperiment
import haiku as hk
import jax
import jax.numpy as jnp
from loguru import logger
import optax
from rich.console import Console
from torch.utils.data import DataLoader

from svd_plus_plus.datasets.datapipe import get_stats
from svd_plus_plus.model.metrics import Loss, RMSEMetric
from svd_plus_plus.model.typing import Batch


class SvdOutput(NamedTuple):
    scores: jnp.ndarray
    items: jnp.ndarray


class BatchOutput(NamedTuple):
    loss: jnp.ndarray
    output: dict[str, jnp.ndarray]
    state: Optional[hk.State] = None


class SvdRunner(IExperiment):
    def __init__(
        self,
        model: hk.Module,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        stats_path: str,
        num_epochs: int,
        seed: int = 13,
        callbacks: dict[str, ICallback] = None,
    ) -> None:
        super().__init__()
        self.state: hk.State = None
        self.rng_seq = hk.PRNGSequence(seed)
        self.forward = hk.transform(self._forward_fn)
        self._console = Console()
        self._stats = get_stats(Path(stats_path))
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._metrics = {"loss": Loss(), "rmse": RMSEMetric()}
        # Set parameters for IExperiment
        self.callbacks = callbacks or {}
        self.seed = seed
        self.num_epochs = num_epochs

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

    @partial(jax.jit, static_argnums=0)
    def init_state(self, rng_key: jax.random.PRNGKey, batch: Batch) -> dict[str, Any]:
        params = self.forward.init(rng_key, batch)
        opt_state = self._optimizer.init(params)
        return {"opt_state": opt_state, "params": params}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        rng_key, batch = next(self.rng_seq), next(iter(self.datasets["train"]))
        self.state = self.init_state(rng_key, batch)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        if self.test_dataset is not None:
            self.dataset_key, self.dataset = "test", self.test_dataset
            self.is_train_dataset = False
            self.run_dataset()

    def on_dataset_start(self, exp: "IExperiment") -> None:
        super().on_dataset_start(exp)
        self._console.print(
            f":rocket: Starting {self.dataset_key} on epoch {self.epoch_step}. "
            "Get some :coffee:. It's going to be fun!!!",
            style="bold green",
        )

    def on_dataset_end(self, exp: "IExperiment") -> None:
        super().on_dataset_end(exp)
        logger.info(f"{self.dataset_key.capitalize()} metrics:")
        max_length = max(len(x) for x in self.dataset_metrics)
        # Sort by length to make it prettier
        for metric in sorted(self.dataset_metrics, key=lambda x: (len(x), x)):
            metric_value = self.dataset_metrics.get(metric)
            # Log metrics that are numbers of some kind only
            if isinstance(metric_value, (float, int)):
                logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")

    def on_batch_end(self, exp: "IExperiment") -> None:
        super().on_batch_end(exp)
        if self.is_train_dataset:
            self.state = self.batch_output.state
        # Update metrics
        for metric in self._metrics.values():
            metric(self.batch_output.loss)
        self.batch_metrics = {key: metric.get_metric() for key, metric in self._metrics.items()}

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

    @partial(jax.jit, static_argnums=(0, 2))
    def predict(self, batch: Batch, k: int = 10) -> SvdOutput:
        output_dict = self.forward.apply(self.state["params"], jax.random.PRNGKey(self.seed), batch)
        return SvdOutput(*jax.lax.top_k(output_dict["output"], k=k))

    def run_batch(self) -> None:
        self.batch_output = (
            self._train_batch(self.state, next(self.rng_seq), self.batch)
            if self.is_train_dataset
            else self._eval_batch(self.state, next(self.rng_seq), self.batch)
        )

    def run_dataset(self) -> None:
        with alive_bar(
            title=f"Iterating {self.dataset_key.capitalize()}",
            total=len(self.dataset),
            # Lower number of chars for progress bar.
            length=20,
        ) as pbar:
            for self.batch in self.dataset:
                self._run_event("on_batch_start")
                self.run_batch()
                self._run_event("on_batch_end")
                # Show in progress bar
                pbar.text = (
                    f"[ {', '.join(f'{k}: {v:.4f}' for k, v in self.batch_metrics.items())} ]"
                )
                pbar()
        self.dataset_metrics = {
            key: metric.get_metric(reset=True) for key, metric in self._metrics.items()
        }

    def run(self, datasets: dict[str, DataLoader]) -> None:
        self.test_dataset = datasets.pop("test", None)
        self.datasets = datasets
        super().run()
