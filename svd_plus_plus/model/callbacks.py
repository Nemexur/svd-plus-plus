from animus import ICallback, IExperiment
from loguru import logger
from rich.console import Console
from tqdm import tqdm

from svd_plus_plus.model.metrics import Metric
import wandb


class MetricsCallback(ICallback):
    def __init__(self, metrics: dict[str, Metric]) -> None:
        super().__init__()
        self._metrics = metrics

    def on_dataset_start(self, exp: "IExperiment") -> None:
        for metric in self._metrics.values():
            metric.reset()

    def on_batch_end(self, exp: "IExperiment") -> None:
        for metric in self._metrics.values():
            metric(exp.batch_output.loss)
        exp.batch_metrics = {key: metric.get_metric() for key, metric in self._metrics.items()}

    def on_dataset_end(self, exp: "IExperiment") -> None:
        exp.dataset_metrics = {
            key: metric.get_metric(reset=True) for key, metric in self._metrics.items()
        }


class ProgressBarCallback(ICallback):
    def __init__(self) -> None:
        super().__init__()
        self._pbar: tqdm = None

    def on_dataset_start(self, exp: "IExperiment") -> None:
        self._pbar = tqdm(
            desc=f"\033[00;33mIterating {exp.dataset_key.capitalize()}\033[0m",
            total=len(exp.dataset),
            bar_format=(
                "{desc} [{n_fmt}/{total_fmt}] "
                "{percentage:3.0f}%|{bar}|{postfix} "
                "({elapsed}<{remaining}, {rate_fmt})"
            ),
        )

    def on_batch_end(self, exp: "IExperiment") -> None:
        self._pbar.set_postfix_str(
            f"[ {', '.join(f'{k}: {v:.4f}' for k, v in exp.batch_metrics.items())} ]"
        )
        self._pbar.update()

    def on_dataset_end(self, exp: "IExperiment") -> None:
        self._pbar.clear()
        self._pbar.close()
        self._pbar = None

    def on_exception(self, exp: "IExperiment") -> None:
        ex = exp.exception
        if not ((ex is not None) and isinstance(ex, BaseException)):
            return
        if isinstance(ex, KeyboardInterrupt):
            if self._pbar is not None:
                self._pbar.write("Keyboard Interrupt")
                self._pbar.clear()
                self._pbar.close()
                self._pbar = None


class LoggerCallback(ICallback):
    def __init__(self) -> None:
        super().__init__()
        self._console = Console()

    def on_epoch_start(self, exp: "IExperiment") -> None:
        self._console.print(
            f":rocket: Starting epoch {exp.epoch_step}. "
            "Get some :coffee:. It's going to be fun!!!",
            style="bold violet",
        )

    def on_dataset_end(self, exp: "IExperiment") -> None:
        logger.info(f"{exp.dataset_key.capitalize()} metrics:")
        max_length = max(len(x) for x in exp.dataset_metrics)
        # Sort by length to make it prettier
        for metric in sorted(exp.dataset_metrics, key=lambda x: (len(x), x)):
            metric_value = exp.dataset_metrics.get(metric)
            if isinstance(metric_value, (float, int)):
                logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")
        if wandb.run is not None:
            wandb.log(
                {"epoch": exp.epoch_step}
                | {
                    f"{exp.dataset_key}/{metric}": value
                    for metric, value in exp.dataset_metrics.items()
                }
            )
