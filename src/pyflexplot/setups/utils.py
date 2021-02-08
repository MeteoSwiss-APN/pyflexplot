"""Some utilities for subpackage ``pyflexplot.setups``."""
# Standard library
from typing import Any
from typing import List

# First-party
from srutils.format import sfmt


def setup_repr(obj: Any) -> str:
    s_attrs_lst: List[str] = []
    for param in obj.get_params():
        s_value = sfmt(getattr(obj, param))
        if "\n" in s_value:
            s_value = s_value.replace("\n", "\n  ")
        s_attrs_lst.append(f"{param}={s_value}")
    s_attrs = ",\n  ".join(s_attrs_lst)
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"
