from typing import Any

from loguru import logger

from svd_plus_plus.registry import Registry


def do_train(config: dict[str, Any]) -> None:
    config = Registry.get_from_params(**config)
    logger.info("Get DataPipes.")
    datapipes = config["dataset"].get_pipes(splits=("train", "valid", "test"))
    logger.info("Configure DataLoaders.")
    datasets = {key: value(dataset=datapipes[key]) for key, value in config["dataloaders"].items()}
    logger.info("Run Experiment.")
    config["experiment"].run(datasets=datasets)
