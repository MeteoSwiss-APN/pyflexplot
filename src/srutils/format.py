"""Format objects as strings."""
# Standard library
from typing import Any
from typing import Collection
from typing import List
from typing import Optional
from typing import Union


def sfmt(obj: Any, q: str = "'", *, none: str = "None") -> str:
    """Format a object to a string, while quoting strings and so forth."""
    if obj is None:
        return none
    elif isinstance(obj, str):
        return f"{q}{obj}{q}"
    else:
        return str(obj)


def capitalize(s, preserve=True):
    """Capitalize a word, optionally preserving uppercase letters.

    Args:
        s (str): String.

        preserve (bool, optional): Whether to preserve capitalized letters.
            Defaults to True.

    """
    s = str(s)
    if not preserve:
        s = s.lower()
    return f"{s[0].upper()}{s[1:]}"


def titlecase(s, preserve=True):
    """Convert a string to titlecase.

    Args:
        s (str): String.

        preserve (bool, optional): Whether to preserve capitalized letters.
            Defaults to True.

    """
    # Words not to be capitalized
    # src: https://stackoverflow.com/a/35018815
    lower = [
        "the",
        "a",
        "an",
        "as",
        "at",
        "but",
        "by",
        "for",
        "in",
        "of",
        "off",
        "on",
        "per",
        "to",
        "up",
        "via",
        "and",
        "nor",
        "or",
        "so",
        "yet",
    ]
    # Note: This is a rather simplistic implementation.
    # A more sophisticated implementation could be guided by, for instance:
    # https://titlecaseconverter.com/words-to-capitalize/?style=AP,APA,CMOS,MLA,NYT,WP

    s = str(s)
    if not preserve:
        s = s.lower()
    words_input = s.split(" ")
    words_title = [capitalize(words_input[0])] + [
        w if w in lower else capitalize(w) for w in words_input[1:-1]
    ]
    if len(words_input) >= 2:
        words_title += [capitalize(words_input[-1])]

    return " ".join(words_title)


def ordinal(i: Union[int, str]) -> str:
    """Format an integer as an ordinal number."""
    i = int(i)
    if abs(i) % 10 == 1:
        sfx = {11: "th"}.get(abs(i) % 100, "st")
    elif abs(i) % 10 == 2:
        sfx = {12: "th"}.get(abs(i) % 100, "nd")
    elif abs(i) % 10 == 3:
        sfx = {13: "th"}.get(abs(i) % 100, "rd")
    else:
        sfx = "th"
    return f"{i}{sfx}"


def nested_repr(obj: Any) -> str:
    """Format (optionally) nested object representations on multiple lines.

    Example:
        ClassA(
            param1: 1,
            b: ClassB(
                param2: "2",
            ),
        )

    """
    s_attrs_lst: List[str] = []
    for param in obj.get_params():
        s_value = sfmt(getattr(obj, param))
        if "\n" in s_value:
            s_value = s_value.replace("\n", "\n  ")
        s_attrs_lst.append(f"{param}={s_value}")
    s_attrs = ",\n  ".join(s_attrs_lst)
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"


# pylint: disable=R0913  # too-many-arguments
# pylint: disable=R0914  # too-many-locals
def format_numbers_range(
    numbers: Collection[float],
    fmt: str = "g",
    delta: float = 1,
    join_range: str = "-",
    join_others: str = ",",
    range_min: int = 3,
) -> str:
    """Format numbers individually or, if they are consecutive, as a range.

    Args:
        numbers: Collection of numbers to be formatted.

        fmt (optional): How to format individual numbers, e.g., "03d" for zero-
            padded three-digit integers.

        delta (optional): Differences between consecutive numbers.

        join_range (optional): Character(s) used to join the first and last
            number in a range.

        join_others (optional): Character(s) used to join non-consecutive
            numbers or ranges.

        range_min (optional): Minimum number of consecutive numbers to be
            formatted as a range.

    """
    template = f"{{num:{fmt}}}"
    consecutive_numbers_groups: List[List[float]] = []
    previous_number: Optional[float] = None
    for number in sorted(numbers):
        if previous_number is None or number - previous_number != delta:
            consecutive_numbers_groups.append([])
        consecutive_numbers_groups[-1].append(number)
        previous_number = number
    formatted_numbers_and_ranges: List[str] = []
    for consecutive_numbers in consecutive_numbers_groups:
        if len(consecutive_numbers) < range_min:
            numbers_to_format = consecutive_numbers
            join = join_others
        else:
            numbers_to_format = [consecutive_numbers[0], consecutive_numbers[-1]]
            join = join_range
        formatted_numbers: List[str] = []
        for number in numbers_to_format:
            try:
                formatted_numbers.append(template.format(num=number))
            except ValueError as e:
                raise Exception(
                    "cannot format number with template", number, template
                ) from e
        formatted_numbers_and_ranges.append(join.join(formatted_numbers))
    return join_others.join(formatted_numbers_and_ranges)
