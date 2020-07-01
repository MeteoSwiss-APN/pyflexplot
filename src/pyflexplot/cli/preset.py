# -*- coding: utf-8 -*-
"""
Preset setup files.
"""
# Standard library
import re
from pathlib import Path
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Pattern
from typing import Union

# Local
from .. import check_dir_exists
from ..utils.exceptions import NoPresetFileFoundError

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
            if file_path.name.startswith("_"):
                continue
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
