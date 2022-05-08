from typing import Any, Callable, List
from functools import partial
from pathlib import Path

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from svd_plus_plus.cli.types import CommaSeparatedTextType, CudaDevicesType
from svd_plus_plus.core import State

cli_group = partial(
    click.group, cls=HelpColorsGroup, help_headers_color="yellow", help_options_color="magenta"
)


cli_command = partial(
    click.command, cls=HelpColorsCommand, help_headers_color="yellow", help_options_color="magenta"
)


pass_state = click.make_pass_decorator(State, ensure=True)


def seed_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        state.core.seed = value
        return value

    return click.option(
        "--seed",
        callback=callback,
        expose_value=False,
        required=False,
        help="Fix seed for the run.",
        type=click.INT,
        show_default=True,
        default=13,
    )(f)


def cuda_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        state.core.cuda = value
        return value

    return click.option(
        "--cuda",
        callback=callback,
        expose_value=False,
        required=False,
        type=CudaDevicesType(),
        help=(
            "CUDA Devices to train model on in format: {gpu_idx},{gpu_idx}. "
            "Example: 0,1,2 means training model on 3 gpus."
        ),
        show_default=True,
        default=-1,
    )(f)


def debug_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        state.core.debug = value
        return value

    return click.option(
        "--debug",
        callback=callback,
        expose_value=False,
        required=False,
        help="Run the command in debug mode.",
        is_flag=True,
    )(f)


def directory_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        state.dir_state.directory = value
        return value

    return click.option(
        "-s",
        "--serialization-dir",
        required=True,
        help="Directory to save files generated with invoked command.",
        callback=callback,
        expose_value=False,
        type=click.Path(path_type=Path),
    )(f)


def force_delete_dir_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        state.dir_state.force_delete = value
        return value

    return click.option(
        "--force-delete",
        is_flag=True,
        help="Force delete directory for command if exists.",
        callback=callback,
        expose_value=False,
    )(f)


def add_extra_info_to_dir_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: List[Any]) -> Any:
        state = ctx.ensure_object(State)
        state.dir_state.extra_info = value
        return value

    return click.option(
        "-da",
        "--dir-add",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        multiple=True,
        help="Additional directories to add in command directory.",
        callback=callback,
        expose_value=False,
    )(f)


def use_wandb_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        state.wandb_state.use_wandb = value
        return value

    return click.option(
        "--use-wandb",
        is_flag=True,
        help="Whether to log to W&B or not.",
        callback=callback,
        expose_value=False,
    )(f)


def wandb_tags_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: List[Any]) -> Any:
        state = ctx.ensure_object(State)
        state.wandb_state.wandb_tags = value
        return value

    return click.option(
        "--wandb-tags",
        type=CommaSeparatedTextType(),
        help="Tags to identify experiment in W&B.",
        callback=callback,
        expose_value=False,
    )(f)


def wandb_project_option(f: Callable) -> Callable:
    def callback(ctx: click.Context, param: click.core.Parameter, value: List[Any]) -> Any:
        state = ctx.ensure_object(State)
        state.wandb_state.wandb_project = value
        return value

    return click.option(
        "--wandb-project",
        envvar="WANDB_PROJECT_NAME",
        type=click.STRING,
        help="W&B project to log experiments. By default looks up WANDB_PROJECT_NAME env.",
        callback=callback,
        expose_value=False,
    )(f)


def default_options(f: Callable) -> Callable:
    f = seed_option(f)
    f = debug_option(f)
    return f


def directory_options(f: Callable) -> Callable:
    f = add_extra_info_to_dir_option(f)
    f = force_delete_dir_option(f)
    f = directory_option(f)
    return f


def wandb_options(f: Callable) -> Callable:
    f = wandb_tags_option(f)
    f = wandb_project_option(f)
    f = use_wandb_option(f)
    return f
