from typing import Any, Callable, Optional
import re

import click


class CudaDevicesType(click.types.ParamType):
    """Comma separated list of arguments for option."""

    @property
    def name(self) -> str:
        return "CUDA_DEVICES"

    def convert(
        self,
        value: str,
        param: Optional[click.core.Parameter],
        ctx: Optional[click.core.Context],
    ) -> Any:
        cuda = super().convert(value=value, param=param, ctx=ctx)
        return (
            [-1]
            if cuda == "-1" or cuda == -1
            else [int(idx) for idx in re.findall(r"(?:\d+|-\d+)", cuda, flags=re.I)]
        )


class DictParamType(click.types.ParamType):
    """A Click type to represent dictionary as parameters for command."""

    def __init__(self, value_type: Callable = str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value_type = value_type

    @property
    def name(self) -> str:
        return "DICT"

    def convert(
        self,
        value: str,
        param: Optional[click.core.Parameter],
        ctx: Optional[click.core.Context],
    ) -> Any:
        extra_vars = super().convert(value=value, param=param, ctx=ctx)
        regex = r"([a-z0-9\_\-\.\+\\\/]+)=([a-z0-9\_\-\.\+\\\/]+)"
        return (
            {
                param: self._value_type(value)
                for param, value in re.findall(regex, extra_vars, flags=re.I)
            }
            if extra_vars is not None
            else None
        )


class CommaSeparatedTextType(click.types.ParamType):
    """A click type to represent comma separated values."""

    @property
    def name(self) -> str:
        return "LIST"

    def convert(
        self,
        value: str,
        param: Optional[click.core.Parameter],
        ctx: Optional[click.core.Context],
    ) -> Any:
        text_info = super().convert(value=value, param=param, ctx=ctx)
        regex = r"[a-z0-9\_\-\.\+\\\/\s]+"
        return re.findall(regex, text_info, flags=re.I) if text_info is not None else None
