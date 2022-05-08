# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The following utility functions are copied from torchtext
# https://github.com/pytorch/text/blob/main/torchtext/data/datasets_utils.py

import functools
import inspect


def _check_default_set(split, target_select, dataset_name):
    # Check whether given object split is either a tuple of strings or string
    # and represents a valid selection of options given by the tuple of strings
    # target_select.
    if isinstance(split, str):
        split = (split,)
    if isinstance(target_select, str):
        target_select = (target_select,)
    if not isinstance(split, tuple):
        raise ValueError("Internal error: Expected split to be of type tuple.")
    if not set(split).issubset(set(target_select)):
        raise TypeError(
            "Given selection {} of splits is not supported for dataset {}. Please choose from {}.".format(
                split, dataset_name, target_select
            )
        )
    return split


def _wrap_datasets(datasets, split):
    # Wrap return value for _setup_datasets functions to support singular values instead
    # of tuples when split is a string.
    if isinstance(split, str):
        if len(datasets) != 1:
            raise ValueError("Internal error: Expected number of datasets is not 1.")
        return datasets[split]
    return datasets


def _wrap_split_argument_with_fn(fn, splits):
    """
    Wraps given function of specific signature to extend behavior of split
    to support individual strings. The given function is expected to have a split
    kwarg that accepts tuples of strings, e.g. ('train', 'valid') and the returned
    function will have a split argument that also accepts strings, e.g. 'train', which
    are then turned single entry tuples. Furthermore, the return value of the wrapped
    function is unpacked if split is only a single string to enable behavior such as
    """
    argspec = inspect.getfullargspec(fn)
    if not (
        argspec.args[1] == "split"
        and argspec.varargs is None
        and argspec.varkw is None
        and len(argspec.kwonlyargs) == 0
    ):
        raise ValueError(
            f"Internal Error: Given function {fn} did not adhere to standard signature."
        )

    @functools.wraps(fn)
    def new_fn(cls, splits=splits, **kwargs):
        result = {}
        for item in _check_default_set(splits, splits, fn.__name__):
            result[item] = fn(cls, split=item, **kwargs)
        return _wrap_datasets(result, splits)

    new_sig = inspect.signature(new_fn)
    new_sig_params = new_sig.parameters
    new_params = []
    new_params.append(new_sig_params["split"].replace(default=splits))
    new_params += [entry[1] for entry in list(new_sig_params.items())[2:]]
    new_sig = new_sig.replace(parameters=tuple(new_params))
    new_fn.__signature__ = new_sig

    return new_fn


def _wrap_split_argument(splits):
    def new_fn(fn):
        return _wrap_split_argument_with_fn(fn, splits)

    return new_fn
