# -*- coding: utf-8 -*-
"""
Summarize objects as a dict for testing etc.
"""
# Standard library
from dataclasses import is_dataclass
from functools import partial
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Union

# Third-party
from pydantic import BaseModel

# First-party
from srutils.iter import isiterable


class NotSummarizableError(Exception):
    """Object could not be summarized."""


class AttributeConflictError(Exception):
    """Conflicting object attributes."""


def default_summarize(
    self: Any,
    addl: Optional[Collection[str]] = None,
    skip: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """Default summarize method; see docstring of ``summarizable``.

    Args:
        self: The class instance to be summarized.

        addl: Additional attributes to be summarized. Added to those specified
            in ``self.summarizable_attrs``.

        skip: Attributes not to be summarized despite being specified in
            ``self.summarizable_attrs``.

    Return:
        Summary dict.

    """
    return Summarizer().run(self, addl=addl, skip=skip)


# pylint: disable=W0613  # unused-argument (self)
def default_post_summarize(
    self: Any, summary: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Default post_summarize method; see docstring of ``summarizable``.

    Args:
        self: The class instance to be summarized.

        summary: Summary dict to be modified.

    Return:
        Modified summary dict.

    """
    return summary


def summarizable(
    cls: Optional[Callable] = None,
    *,
    attrs: Optional[Collection[str]] = None,
    attrs_skip: Optional[Collection[str]] = None,
    summarize: Optional[
        Callable[[Any, MutableMapping[str, Any]], MutableMapping[str, Any]]
    ] = None,
    post_summarize: Optional[
        Callable[[Any, MutableMapping[str, Any]], MutableMapping[str, Any]]
    ] = None,
    auto_collect: bool = True,
    overwrite: bool = False,
) -> Callable:
    """Decorator to make a class summarizable.

    Args:
        cls: Class to be decorated.

        attrs: Class attributes to summarize. Added to ``cls`` as class
            attribute ``summarizable_attrs``, which also contains all auto-
            collected attributes of special classes like dataclasse (unless
            ``auto_collect`` is False), but none specified in ``attrs_skip``.

        attrs_skip: Class attributes not to summarize. Affects attributes in
            ``attrs`` and, more importantly, auto-collected ones such as
            dataclass attributes.

        summarize: Custom function to summarize the class. Returns a dict
            containing the summarized attributes, which is then used to update
            the existing summary dict that has been created based on ``attrs``.
            Replaces ``default_summarize``. Added to ``cls`` as method
            ``summarize``.

        post_summarize: Custom function to post-process the summary dict.
            Replaces ``default_post_summarize``. Added to ``cls`` as method
            ``post_summarize``.

        auto_collect: Auto-collect attributes of certain types of classes, such
            as data classes and dict-convertible classes, in addition to those
            specified in attrs (if given at all).

        overwrite: Overwrite existing class attributes and/or methods. Must be
            True to make classes summarizable that inherit from summarizable
            parent classes.

    """
    if cls is None:
        return partial(
            summarizable,
            attrs=attrs,
            attrs_skip=attrs_skip,
            summarize=summarize,
            post_summarize=post_summarize,
            auto_collect=auto_collect,
            overwrite=overwrite,
        )

    try:
        attrs = list(attrs or [])
    except TypeError:
        raise ValueError("`attrs` is not iterable", type(attrs), attrs)
    try:
        attrs_skip = list(attrs_skip or [])
    except TypeError:
        raise ValueError("`attrs_skip` is not iterable", type(attrs_skip), attrs_skip)
    if summarize is None:
        summarize = default_summarize
    if post_summarize is None:
        post_summarize = default_post_summarize

    if auto_collect:
        if is_attrs_class(cls):
            # Collect attributes defined with ``attr.attrib``
            attrs = [a.name for a in cls.__attrs_attrs__] + attrs  # type: ignore
        elif is_dataclass(cls):
            # Collect dataclass fields
            attrs = list(cls.__dataclass_fields__) + attrs  # type: ignore
        elif issubclass(cls, BaseModel):  # type: ignore
            # Collect model fields
            attrs = list(cls.__fields__) + attrs  # type: ignore
    attrs = [a for a in attrs if a not in attrs_skip]

    # Extend class
    for name, attr in [
        ("summarizable_attrs", attrs),
        ("summarize", summarize),
        ("post_summarize", post_summarize),
    ]:
        if not overwrite and hasattr(cls, name):
            raise AttributeConflictError(name, cls)
        setattr(cls, name, attr)
    return cls


class Summarizer:
    """Summarize an as a dict.

    Subclasses must define the property ``summarizable_attrs``, comprising
    a list of attribute names to be collected.

    If attribute values possess a ``summarize`` method themselves, the output
    of that is collected. Otherwise, it is attemted to convert the values to
    common types like dicts or lists. If all attempts fail, the raw value is
    added to the summary dict.

    """

    def run(
        self,
        obj: Any,
        *,
        addl: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
    ) -> Dict[str, Any]:
        """Summarize specified attributes of ``obj`` in a dict.

        The attributes to be summarized must be specified by name in the
        attribute ``obj.summarizable_attrs``.

        Args:
            obj: Object to summarize.

            addl (optional): Additional attributes to be collected.

            skip (optional): Attributes to skip during collection.

        Returns:
            Dictionary containing the collected attributes and their values.

        """
        data: Dict[str, Any] = {}

        if skip is None or "type" not in skip:
            data["type"] = type(obj).__name__

        try:
            attrs = list(obj.summarizable_attrs)
        except AttributeError:
            if addl is not None or skip is not None:
                raise ValueError(
                    "arguments addl and skip invalid without obj.summarizable_attrs"
                )
            return self._summarize(obj)

        if addl is not None:
            attrs += [a for a in addl if a not in attrs]
        if skip is not None:
            attrs = [a for a in attrs if a not in skip]
        for attr in attrs:
            data[attr] = self._summarize(getattr(obj, attr))
        return obj.post_summarize(data)

    def _summarize(self, obj: Any) -> Union[Dict[str, Any], Any]:
        """Try to summarize the object in various ways."""
        methods = [
            self._try_summarizable,
            self._try_dict_like,
            self._try_list_like,
            self._try_named,
        ]
        for method in methods:
            try:
                return method(obj)
            except NotSummarizableError:
                continue
        return obj

    def _try_summarizable(self, obj: Any) -> Dict[str, Any]:
        """Try to summarize ``obj`` as a summarizable object."""
        try:
            data = obj.summarize()
        except AttributeError:
            raise NotSummarizableError("summarizable", obj)
        else:
            return self._summarize(data)

    def _try_dict_like(self, obj: Any) -> Dict[Any, Any]:
        """Try to summarize ``obj`` as a dict-like object."""
        try:
            items = obj.items()
        except AttributeError:
            try:
                obj = dict(obj)
            except (TypeError, ValueError):
                raise NotSummarizableError("dict-like", obj)
            else:
                items = obj.items()
        return {self._summarize(key): self._summarize(val) for key, val in items}

    def _try_list_like(self, obj: Any) -> Sequence[Any]:
        """Try to summarize ``obj`` as a list-like object."""
        if not isiterable(obj, str_ok=False):
            raise NotSummarizableError("list-like", obj)
        data = []
        for item in obj:
            data.append(self._summarize(item))
        return data

    def _try_named(self, obj: Any) -> str:
        """Try to summarize ``obj`` as a named object (e.g., function/method)."""
        try:
            name = obj.__name__
        except AttributeError:
            pass
        else:
            try:
                obj_self = obj.__self__
            except AttributeError:
                pass
            else:
                name = f"{obj_self.__class__.__name__}.{name}"
            return f"{type(obj).__name__}:{name}"
        raise NotSummarizableError("named", obj)


def is_attrs_class(cls: Any) -> bool:
    """Determine whether a class has been defined with ``@attr.attrs``."""
    return isinstance(cls, type) and hasattr(cls, "__attrs_attrs__")