# -*- coding: utf-8 -*-
"""
Preset setup files.
"""
# Standard library
import re
import sys
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Union

# Third-party
import click
from click import Context

# Local
from . import check_dir_exists
from .exceptions import NoPresetFileFoundError
from .logging import log
from .typing import ClickParamType

preset_paths: List[Union[str, Path]] = []


def add_to_preset_paths(path: Union[Path, str], first: bool = True) -> None:
    """Add a path to the preset path list.

    Args:
        path: Path to preset files.

        first (optional): Add the new path ahead of existing ones.

    """
    global preset_paths  # pylint: disable=W0603  # global-statement
    path = Path(path)
    check_dir_exists(path)
    idx = 0 if first else -1
    preset_paths.insert(idx, path)


def collect_preset_paths() -> Iterator[Path]:
    """Collect all setup file paths as specified in ``preset_paths``."""
    global preset_paths  # pylint: disable=W0603  # global-statement
    for path in preset_paths:
        check_dir_exists(path)
        yield Path(path)


def compile_patterns(patterns: Collection[str]) -> List[Pattern]:
    rx_patterns = []
    for pattern in patterns:
        ch = "[a-zA-Z0-9_.-/]"
        rx_pattern = re.compile(
            r"\A" + pattern.replace("*", f"{ch}*").replace("?", ch) + r"\Z"
        )
        rx_patterns.append(rx_pattern)
    return rx_patterns


def collect_preset_files(
    patterns: Collection[str] = "*", antipatterns: Optional[Collection[str]] = None,
) -> Dict[Path, Dict[str, Path]]:
    """Collect all setup files in locations specified in ``preset_paths``."""
    rx_patterns = compile_patterns(patterns)
    rx_antipatterns = [] if antipatterns is None else compile_patterns(antipatterns)
    files_by_preset_path = {}  # type: ignore
    for preset_path in collect_preset_paths():
        files_by_preset_path[preset_path] = {}
        for file_path in sorted(preset_path.rglob("*.toml")):
            file_path_rel = file_path.relative_to(preset_path)
            name = str(file_path_rel)[: -len(file_path.suffix)]
            for rx in rx_antipatterns:
                if rx.match(name):
                    break
            for rx in rx_patterns:
                if rx.match(name):
                    files_by_preset_path[preset_path][name] = file_path
                    break
        if not files_by_preset_path[preset_path]:
            raise NoPresetFileFoundError(patterns, preset_path)
    return files_by_preset_path


# pylint: disable=W0613  # unused-argument (ctx, param)
def click_add_to_preset_paths(ctx: Context, param: ClickParamType, value: Any) -> None:
    if not value:
        return
    add_to_preset_paths(value)


def collect_preset_files_flat(pattern: str) -> Dict[str, Path]:
    files_by_dir = collect_preset_files([pattern])
    named_paths = {
        name: path for files in files_by_dir.values() for name, path in files.items()
    }
    if not named_paths:
        raise NoPresetFileFoundError(pattern, files_by_dir)
    return named_paths


def cat_preset(name: str, include_source: bool = False) -> str:
    """Print the content of a preset setup file and exit."""
    lines = []
    for path in collect_preset_files_flat(name).values():
        if include_source:
            lines.append(f"# source: {path}\n")
        with open(path) as f:
            lines.extend([l.strip() for l in f.readlines()])
    return "\n".join(lines)


def click_list_presets_and_exit(
    ctx: Context, param: ClickParamType, value: Any
) -> None:
    """List all presets setup files and exit."""
    if not value:
        return
    click_find_presets_and_exit(ctx, param, "*")


# pylint: disable=W0613  # unused-argument (param)
def click_find_presets_and_exit(
    ctx: Context, param: ClickParamType, value: Any
) -> None:
    """Find preset setup file(s) by name (optional wildcards) and exit."""
    if not value:
        return
    assert isinstance(value, Sequence)  # mypy
    assert isinstance(value[0], str)  # mypy
    _click_list_presets(ctx, collect_preset_files(value))
    ctx.exit(0)


# pylint: disable=W0613  # unused-argument (param)
def click_cat_preset_and_exit(ctx: Context, param: ClickParamType, value: Any) -> None:
    """Print the content of a preset setup file and exit."""
    if not value:
        return
    verbosity = ctx.obj["verbosity"]
    try:
        content = cat_preset(value, include_source=(verbosity > 0))
    except NoPresetFileFoundError:
        click.echo(f"Error: preset '{value}' not found.", file=sys.stderr)
        ctx.exit(1)
    else:
        click.echo(content)
        ctx.exit(0)


# pylint: disable=W0613  # unused-argument (param)
def click_use_preset(ctx: Context, param: ClickParamType, value: Any) -> None:
    if not value:
        return

    if value == ("?",):
        click.echo("Available presets ('?'):")
        click_find_presets_and_exit(ctx, param, "*")

    patterns: Sequence[str] = value
    antipatterns: Sequence[str] = ctx.params.get("preset_skip", [])

    key = "preset_setup_file_paths"
    if key not in ctx.obj:
        ctx.obj[key] = []

    for pattern in patterns:
        try:
            files_by_preset_path = collect_preset_files([pattern], antipatterns)
        except NoPresetFileFoundError:
            click.echo(
                f"Error: No preset setup file found for '{pattern}'.", file=sys.stderr,
            )
            _click_propose_alternatives(pattern)
            ctx.exit(1)
        else:
            n = sum([len(files) for files in files_by_preset_path.values()])
            if n == 0:
                log(vbs="Collected no preset setup files")
            elif n == 1:
                log(vbs=f"Collected {n} preset setup file:")
            else:
                log(vbs=f"Collected {n} preset setup files:")
            _click_list_presets(ctx, files_by_preset_path, indent_all=True)
        for files in files_by_preset_path.values():
            for path in files.values():
                if path not in ctx.obj[key]:
                    ctx.obj[key].append(path)


def _click_list_presets(
    ctx: Context,
    files_by_preset_path: Mapping[Path, Mapping[str, Path]],
    indent_all: bool = False,
) -> None:
    for preset_path, files in files_by_preset_path.items():
        log(vbs=f"{preset_path}:")
        for name, path in files.items():
            log(
                vbs=f"{'  ' if indent_all else ''} name", dbg=f"  {name:23}  {path}",
            )


def _click_propose_alternatives(name: str) -> None:
    try:
        alternatives = collect_preset_files_flat(f"*{name}*")
    except NoPresetFileFoundError:
        pass
    else:
        if alternatives:
            click.echo("Are you looking for any of these?", file=sys.stderr)
            click.echo(" " + "\n ".join(alternatives.keys()), file=sys.stderr)
