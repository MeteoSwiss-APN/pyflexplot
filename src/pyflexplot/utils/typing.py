"""Custom types for type hints."""
# Standard library
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
# Third party
import click

# Primary
ColorType = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
ClickParamType = Union[click.Option, click.Parameter]
FontSizeType = Union[str, float]
LocationType = Union[str, int]
MarkerStyleType = Union[str, int]
RawTextBlockType = Union[str, Sequence[str]]
RectType = Tuple[float, float, float, float]
TextBlockType = List[str]

# Secondary
RawTextBlocksType = Union[str, Sequence[RawTextBlockType]]
TextBlocksType = List[TextBlockType]
