from pathlib import Path

import click

from svd_plus_plus.cli.commands.prepare_data.utils import do_prepare_data
from svd_plus_plus.cli.options import cli_command, default_options, directory_options, pass_state
from svd_plus_plus.core import State
from svd_plus_plus.core.utils import seed_everything


@cli_command(help="Prepare Netflix Prize datasets.")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--use-exp-decay", is_flag=True, help="Whether to add exp decay to ratings.")
@click.option(
    "--top-k",
    type=click.INT,
    help="Number of similar items to consider.",
    default=100,
    show_default=True,
)
@directory_options
@default_options
@pass_state
def prepare_data(state: State, directory: Path, use_exp_decay: bool, top_k: int) -> None:
    seed_everything(state.core.seed)
    state.dir_state.confirm_directory_creation_if_needed()
    do_prepare_data(directory, use_exp_decay, top_k, save_dir=state.dir_state.directory)
    click.secho("Finished!", fg="green")


if __name__ == "__main__":
    prepare_data()
