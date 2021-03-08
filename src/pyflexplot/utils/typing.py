"""Custom types for type hints."""
# Standard library
from typing import Callable
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import click
import numpy as np

# Primary
ColorType = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
ClickParamType = Union[click.Option, click.Parameter]
FontSizeType = Union[str, float]
LocationType = Union[str, int]
MarkerStyleType = Union[str, int]
PointConverterT = Union[
    Callable[[float, float], Tuple[float, float]],
    Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
]
RawTextBlockType = Union[str, Sequence[str]]
RectType = Tuple[float, float, float, float]
TextBlockType = List[str]

# Secondary
RawTextBlocksType = Union[str, Sequence[RawTextBlockType]]
TextBlocksType = List[TextBlockType]
