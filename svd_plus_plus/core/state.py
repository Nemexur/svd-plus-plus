from typing import Any
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil

import click
from loguru import logger
import wandb


@dataclass
class DirState:
    directory: Path = None
    force_delete: bool = False
    extra_info: list[Path] = None

    def confirm_directory_creation_if_needed(self) -> None:
        # If directory exists and it is not empty, confirm delete command.
        if self.directory.exists() and any(self.directory.iterdir()):
            confirmed = (
                click.confirm(
                    click.style(
                        f"Directory at path `{self.directory}` already exists. Delete it?",
                        fg="magenta",
                    ),
                    show_default=True,
                    default=True,
                )
                if not self.force_delete
                else True
            )
            if confirmed:
                shutil.rmtree(self.directory)
                logger.success("Directory successfully deleted!")
                self.directory.mkdir(exist_ok=False)
            else:
                logger.debug("Working with current serialization directory.")
        else:
            # Turns out directory is empty. We do not need to create it.
            if not self.directory.exists():
                self.directory.mkdir(exist_ok=False)

    def add_extra_info_if_needed(self) -> None:
        for directory in self.extra_info or []:
            shutil.copytree(directory, self.directory / directory.stem)


@dataclass
class WBState:
    use_wandb: bool = False
    wandb_project: str = None
    wandb_tags: list[str] = None

    def init_with_config(self, config: dict[str, Any]) -> None:
        if not self.use_wandb:
            logger.debug("W&B is disabled for this run.")
            return
        logger.info("Init W&B logger.")
        wandb.init(
            project=self.wandb_project,
            config=config,
            reinit=True,
            tags=self.wandb_tags,
        )


@dataclass
class Core:
    seed: int = 13
    cuda: list[int] = None
    debug: bool = False


@dataclass
class State:
    dir_state: DirState = field(default_factory=DirState)
    wandb_state: WBState = field(default_factory=WBState)
    core: Core = field(default_factory=Core)

    def as_dict(self) -> dict[str, Any]:
        dict_repr = asdict(self)
        dict_repr["dir_state"]["directory"] = str(dict_repr["dir_state"]["directory"])
        return dict_repr
