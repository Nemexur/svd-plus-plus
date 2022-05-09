from functools import lru_cache

from hydra_slayer import Registry as HydraSlayerRegistry


def _datasets(registry: HydraSlayerRegistry) -> None:
    from svd_plus_plus import datasets

    registry.add_from_module(datasets, prefix="Datasets.")


def _models(registry: HydraSlayerRegistry) -> None:
    from svd_plus_plus.model import model

    registry.add_from_module(model, prefix="Models.")


def _losses(registry: HydraSlayerRegistry) -> None:
    from svd_plus_plus.model.losses import mse_loss, warp_loss

    registry.add(mse_loss, name="mse_loss")
    registry.add(warp_loss, name="warp_loss")


def _runners(registry: HydraSlayerRegistry) -> None:
    from svd_plus_plus.model import SvdRunner

    registry.add(SvdRunner, name="Runners.SvdRunner")


@lru_cache()
def get_registry() -> HydraSlayerRegistry:
    registry = HydraSlayerRegistry(name_key="type")
    # Lately adding modules
    registry.late_add(_datasets)
    registry.late_add(_models)
    registry.late_add(_losses)
    registry.late_add(_runners)
    return registry


Registry = get_registry()  # noqa: E221
