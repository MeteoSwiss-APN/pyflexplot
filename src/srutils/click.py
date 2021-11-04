"""Click utilities."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

# Third-party
import click


class CharSepList(click.ParamType):
    """List of elements of a given type separated by a given character."""

    def __init__(
        self,
        type_: Type[Any],
        separator: str,
        *,
        name: Optional[str] = None,
        unique: bool = False,
    ) -> None:
        """Create an instance of ``CharSepList``.

        Args:
            type_: Type of list elements.

            separator: Separator of list elements.

            name (optional): Name of the type. If omitted, the default name is
                derived from ``type_`` and ``separator``.

            unique (optional): Whether the list elements must be unique.

        Example:
            Create type for comma-separated list of (unique) integers:

            > comma_separated_list_of_unique_ints = CharSepList(int, ',')

        """
        if issubclass(type_, float) and separator == ".":
            raise ValueError(
                f"invalid separator '{separator}' for type " f"'{type_.__name__}'"
            )

        self.type_: Type[Any] = type_
        self.separator: str = separator
        self.unique: bool = unique
        self.name: str = name or f"{type_.__name__}{separator}" * 2 + "..."

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> list[Union[str, Type[Any]]]:
        """Convert a string to a list of ``type_`` elements."""
        values_str = value.split(self.separator)
        values: list[Union[str, Type[Any]]] = []
        for i, value_str in enumerate(values_str):
            try:
                value = self.type_(value_str)
            except (ValueError, TypeError) as e:
                self.fail(
                    f"Invalid '{self.separator}'-separated list '{value}': "
                    f"Value '{value_str}' ({i + 1}/{len(values_str)}) "
                    f"incompatible with type '{self.type_.__name__}' "
                    f"({type(e).__name__}: {e})"
                )
            else:
                if self.unique and value in values:
                    n = len(values_str)
                    self.fail(
                        f"Invalid '{self.separator}'-separated list "
                        f"'{value}': Value '{value_str}' ({i + 1}/{n}) "
                        f"not unique"
                    )
                values.append(value)
        return values


class DerivChoice(click.ParamType):
    """Choices from which additional choices can be derived."""

    name: str = "choice"

    def __init__(
        self, base_choices: Sequence[str], derived_choices: Mapping[str, Sequence[str]]
    ) -> None:
        """Create instance of ``DerivChoice``.

        Args:
            base_choices: Base choices.

            derived_choices (dict[str, list[str]]): Derived choices
                constituting combinations of multiple base choices.

        """
        self.base_choices: list[str] = list(base_choices)
        self.derived_choices: dict[str, list[str]] = {
            key: list(val) for key, val in derived_choices.items()
        }
        self._check_derived_choices()

    def get_metavar(self, param: click.Parameter) -> str:
        choices = list(self.base_choices) + list(self.derived_choices)
        return f"[{'|'.join(choices)}]"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        """Check that a string is among the given choices or combinations."""
        if value in self.base_choices:
            return value
        try:
            value = self.derived_choices[value]
        except KeyError:
            choices = self.base_choices + list(self.derived_choices)
            s_choices = ", ".join([f"'{s}'" for s in choices])
            self.fail(f"wrong choice '{value}': must be one of {s_choices}")
        return value

    def _check_derived_choices(self) -> None:
        for name, derived_choice in self.derived_choices.items():
            if name in self.base_choices:
                raise ValueError(
                    f"derived choice '{name}' is already among base choices"
                    f"{self.base_choices}"
                )
            if isinstance(derived_choice, str):
                derived_choice = [derived_choice]
            try:
                it = iter(derived_choice)
            except TypeError as e:
                raise ValueError(
                    "derived choice '{name}' is defined as non-iterable "
                    f"{type(derived_choice).__name__} object: {derived_choice}"
                ) from e
            else:
                for element in it:
                    if element not in self.base_choices:
                        raise ValueError(
                            f"element {element} of derived choice '{name}' element is"
                            f"not among base choices {self.base_choices}"
                        )
