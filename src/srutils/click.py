# -*- coding: utf-8 -*-
"""
Click utilities.
"""
# Standard library
import functools

# Third-party
import click


def click_options(f_options):
    """
    Define a list of click options that can be shared by multiple commands.

    Args:
        f_options (function): Function returning a list of ``click.option``
            objects.

    Example:
        > @click_options          # <= define options
        > def common_options():
        >     return [click.option(...), click.option(...), ...]

        > @click.group
        > def main(...):
        >     ...

        > @CLI.command
        > @common_options         # <= use options
        > def foo(...):
        >     ...

        > @CLI.command
        > @common_options         # <= use options
        > def bar(...):
        >     ...


    Applications:
        Define options that to be shared by multiple commands but are passed
        after the respective command, instead of before as group options (the
        native way to define shared options) are.

        Define options used only by a single group or command in a function
        instead of as decorators, which allows them to be folded by the editor
        more easily.

    Source:
        https://stackoverflow.com/a/52147284

    """
    return lambda f: functools.reduce(lambda x, opt: opt(x), f_options(), f)


class CharSepList(click.ParamType):
    """
    List of elements of a given type separated by a given character.
    """

    def __init__(self, type_, separator, *, name=None, unique=False):
        """Create an instance of ``CharSepList``.

        Args:
            type_ (type): Type of list elements.

            separator (str): Separator of list elements.

            name (str, optional): Name of the type. If omitted, the default
                name is derived from ``type_`` and ``separator``. Defaults to
                None.

            unique (bool, optional): Whether the list elements must be unique.
                Defaults to False.

        Example:
            Create type for comma-separated list of (unique) integers:

            > comma_separated_list_of_unique_ints = CharSepList(int, ',')

        """
        if isinstance(type_, float) and separator == ".":
            raise ValueError(
                f"invalid separator '{separator}' for type " f"'{type_.__name__}'"
            )

        self.type_ = type_
        self.separator = separator
        self.unique = unique
        if name is not None:
            self.name = name
        else:
            s = f"{type_.__name__}{separator}" * 2
            self.name = f"{s}..."

    def convert(self, value, param, ctx):
        """Convert a string to a list of ``type_`` elements."""
        values_str = value.split(self.separator)
        values = []
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
    """
    Choices from which additional choices can be derived.
    """

    name = "choice"

    def __init__(self, base_choices, derived_choices):
        """Create instance of ``DerivChoice``.

        Args:
            base_choices (list[str]): Base choices.

            derived_choices (dict[str, list[str]]): Derived choices
                constituting combinations of multiple base choices.

        """
        self.base_choices = base_choices
        self.derived_choices = derived_choices
        self._check_derived_choices()

    def get_metavar(self, param):
        choices = list(self.base_choices) + list(self.derived_choices)
        return f"[{'|'.join(choices)}]"

    def convert(self, value, param, ctx):
        """Check that a string is among the given choices or combinations."""
        if value in self.base_choices:
            return value
        try:
            return self.derived_choices[value]
        except KeyError:
            choices = self.base_choices + list(self.derived_choices)
            s_choices = ", ".join([f"'{s}'" for s in choices])
            self.fail(f"wrong choice '{value}': must be one of {s_choices}")

    def _check_derived_choices(self):
        for name, derived_choice in self.derived_choices.items():
            if name in self.base_choices:
                raise ValueError(
                    "derived choice is already a base choice",
                    name=name,
                    base_choices=self.base_choices,
                )
            if isinstance(derived_choice, str):
                derived_choice = [derived_choice]
            try:
                it = iter(derived_choice)
            except TypeError:
                raise ValueError(
                    "derived choice is defined as non-iterable "
                    f"{type(derived_choice).__name__} object",
                    name=name,
                    derived_choice=derived_choice,
                )
            else:
                for element in it:
                    if element not in self.base_choices:
                        raise ValueError(
                            "derived choice element is not a base choice",
                            name=name,
                            element=element,
                            base_choices=self.base_choices,
                        )
