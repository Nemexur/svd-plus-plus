from typing import Union
from abc import ABC, abstractmethod

import jax.numpy as jnp


class Metric(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_metric(self, reset: bool = False) -> Union[float, dict[str, float]]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class Loss(Metric):
    def __init__(self) -> None:
        self._loss = self._total = 0.0

    def __call__(self, loss: jnp.ndarray) -> None:
        self._loss += loss.item()
        self._total += 1

    def get_metric(self, reset: bool = False) -> float:
        metric = self._loss / self._total
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        self._loss = self._total = 0.0


class RMSEMetric(Metric):
    def __init__(self) -> None:
        self._rmse = self._total = 0.0

    def __call__(self, mse_loss: jnp.ndarray) -> None:
        self._rmse += jnp.sqrt(mse_loss).item()
        self._total += 1

    def get_metric(self, reset: bool = False) -> float:
        metric = self._rmse / self._total
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        self._rmse = self._total = 0.0
