from pathlib import Path

from animus import set_global_seed
import click
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from loguru import logger
from rich import print_json
import yaml

from svd_plus_plus.cli.commands.train.utils import do_train
from svd_plus_plus.cli.options import (
    cli_command,
    cuda_option,
    default_options,
    directory_options,
    pass_state,
    wandb_options,
)
from svd_plus_plus.cli.types import DictParamType
from svd_plus_plus.core import State
from svd_plus_plus.core.utils import make_flat


@cli_command(help="Train the model with specified config.")
@click.argument("config-path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--extra-vars",
    type=DictParamType(),
    help=(
        "Extra variables to inject in yaml config. "
        "Format: {key_name1}={new_value1},{key_name2}={new_value2},..."
    ),
)
@cuda_option
@directory_options
@default_options
@wandb_options
@pass_state
def train(state: State, config_path: Path, extra_vars: dict[str, str]) -> None:
    state.dir_state.confirm_directory_creation_if_needed()
    # Create config from template
    jinja_env = Environment(loader=FileSystemLoader("."), undefined=StrictUndefined)
    config = yaml.safe_load(
        jinja_env.get_template(str(config_path)).render(**extra_vars, **state.as_dict())
    )
    logger.info("Run with config:")
    print_json(data=config)
    # Preparation for experiment
    set_global_seed(state.core.seed)
    state.wandb_state.init_with_config(make_flat(config))
    # Ok. Let's go.
    do_train(config)


if __name__ == "__main__":
    train()
