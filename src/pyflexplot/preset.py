# -*- coding: utf-8 -*-
"""
Preset setup files.
"""
# Standard library
import re
import sys
from pathlib import Path
from typing import Dict
from typing import Iterator
from typing import List
from typing import Union

# Third-party
import click

# Local
from . import check_dir_exists
from . import data_path

preset_path: List[Union[str, Path]] = []


def add_preset_path(path: Union[Path, str], first=True):
    global preset_path
    path = Path(path)
    check_dir_exists(path)
    idx = 0 if first else -1
    preset_path.insert(idx, path)


add_preset_path(data_path / "presets")


def collect_preset_paths() -> Iterator[Path]:
    """Collect all setup file paths as specified in ``preset_path``."""
    global preset_path
    for path in preset_path:
        check_dir_exists(path)
        yield Path(path)


def collect_preset_files(pattern: str = "*") -> Dict[Path, Dict[str, Path]]:
    """Collect all setup files in locations specified in ``preset_path``."""
    ch = "[a-zA-Z0-9_.-]"
    rx_pattern = re.compile(
        r"\A" + pattern.replace("*", f"{ch}*").replace("?", ch) + r"\Z"
    )
    files_by_dir = {}  # type: ignore
    for dir in collect_preset_paths():
        files_by_dir[dir] = {}
        for path in dir.glob(f"*.toml"):
            name = path.name[: -len(path.suffix)]
            if rx_pattern.match(name):
                files_by_dir[dir][name] = path
    return files_by_dir


def click_add_preset_path(ctx, param, value):
    if not value:
        return
    add_preset_path(value)


def collect_preset_files_flat(name: str):
    files_by_dir = collect_preset_files(name)
    named_paths = {
        name: path for files in files_by_dir.values() for name, path in files.items()
    }
    if not named_paths:
        raise ValueError("preset not found", name, files_by_dir)
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


def click_list_presets(ctx, param, value):
    """List all presets setup files and exit."""
    if not value:
        return
    click_find_presets(ctx, None, "*")


def click_find_presets(ctx, param, value):
    """Find preset setup file(s) by name (optional wildcards) and exit."""
    if not value:
        return
    _click_list_presets(ctx, collect_preset_files(value))
    ctx.exit(0)


def click_cat_preset(ctx, param, value):
    """Print the content of a preset setup file and exit."""
    if not value:
        return
    verbosity = ctx.obj["verbosity"]
    try:
        content = cat_preset(value, include_source=(verbosity > 0))
    except ValueError:
        click.echo(f"Error: preset '{value}' not found!", file=sys.stderr)
        ctx.exit(1)
    else:
        click.echo(content)
        ctx.exit(0)


def click_use_preset(ctx, param, value):
    if not value:
        return
    key = "preset_setup_file_paths"
    if key not in ctx.obj:
        ctx.obj[key] = []
    for name in value:
        try:
            files_by_dir = collect_preset_files(name)
        except ValueError:
            click.echo(
                f"Error: No preset setup file found for '{name}'!", file=sys.stderr,
            )
            _click_propose_alternatives(name)
            ctx.exit(1)
        else:
            n = sum([len(files) for files in files_by_dir.values()])
            click.echo(f"Collected {n} preset setup file{'' if n == 1 else 's'}:")
            _click_list_presets(ctx, files_by_dir, indent_all=True)
        for files in files_by_dir.values():
            for path in files.values():
                if path not in ctx.obj[key]:
                    ctx.obj[key].append(path)


def _click_list_presets(ctx, preset_files_by_dir, indent_all=False):
    verbosity = ctx.obj["verbosity"]
    for dir, files in preset_files_by_dir.items():
        if verbosity > 0:
            click.echo(f"{dir}:")
        for name, path in files.items():
            if verbosity == 0:
                click.echo(f"{'  ' if indent_all else ''}{name}")
            elif verbosity == 1:
                click.echo(f"  {name}")
            else:
                click.echo(f"  {name:23}  {path}")


def _click_propose_alternatives(name):
    try:
        alternatives = collect_preset_files_flat(f"*{name}*")
    except ValueError:
        pass
    else:
        if alternatives:
            click.echo("Are you looking for any of these?", file=sys.stderr)
            click.echo(" " + "\n ".join(alternatives.keys()), file=sys.stderr)
