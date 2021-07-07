"""Some testing utils."""
# Standard library
import dataclasses as dc
from pprint import pformat
from typing import Any
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence

# Third-party
import numpy as np

# Local
from .format import sfmt
from .iter import isiterable


class CheckFailedError(Exception):
    pass


def property_obj(cls, *args, **kwargs):
    """Define a class property creating a given object on-the-fly.

    The purpose of creating the object on-the-fly in a property method is to
    isolate any errors during instatiation in the test methods using the
    property. This prevents the whole test suite from being aborted, as it
    would if the object were defined as a simple class attribute and its
    instatiation failed -- instead, only the tests attempting to use the object
    will fail.

    And yes, this is indeed merely a convenience function to save two lines of
    code wherever it is used. :-)

    Usage:
        The following class definitions are equivalent:

        >>> class C1:
        ...     @property
        ...     def w(self):
        ...         return TranslatedWord(en='train', de='Zug')

        >>> class C2:
        ...     w = property_obj(en='train', de='Zug')

    """
    # pylint: disable=W0613  # unused-argument (self)
    def create_obj(self):
        return cls(*args, **kwargs)

    return property(create_obj)


class IgnoredElement:
    """Element that is ignored in comparisons."""

    def __init__(self, description=None):
        """Create an instance of ``IgnoredElement``."""
        self.description = description

    def __repr__(self):
        return f"{type(self).__name__}({sfmt(self.description)})"


class UnequalElement:
    """Element unequal to any other; useful for force-fail tests."""

    def __init__(self, description=None):
        """Create an instance of ``UnequalElement``."""
        self.description = description

    def __repr__(self):
        return f"{type(self).__name__}({sfmt(self.description)})"

    def __eq__(self, other):
        return False


def ignored(obj):
    return isinstance(obj, IgnoredElement)


def assert_is_sub_element(
    obj_sub: Any, obj_super: Any, name_sub: str = "sub", name_super: str = "super"
) -> None:
    """Check recursively that ``obj_sub`` is a sub-element of ``obj_super``.

    See docstring of ``check_is_sub_element`` for more details on arguments.

    Args:
        obj_sub: Sub-object of ``obj_super``.

        obj_super: Super-object of ``obj_sub``.

        name_sub (optional): Name of ``obj_sub``.

        name_super (optional): Name of ``obj_super``.

    """
    try:
        check_is_sub_element(obj_sub, obj_super, name_sub, name_super)
    except CheckFailedError as e:
        raise AssertionError(str(e)) from e


# pylint: disable=R0912  # too-many-branches (>12)
# pylint: disable=R0913  # too-many-arguments
# pylint: disable=R0914  # too-many-locals
def check_is_sub_element(
    obj_sub: Any,
    obj_super: Any,
    name_sub: str = "sub",
    name_super: str = "super",
    idx_list: Optional[int] = None,
    idx_dict: Optional[int] = None,
):
    """Check recursively that ``obj_sub`` is a sub-element of ``obj_super``.

    Args:
        obj_sub: Any object, but generally a nested structure comprised of
            containers like dicts, lists etc.; all elements in ``obj_sub`` must
            be present in ``obj_super`` at the same nesting level and position.

        obj_super: Like ``obj_sub``; all elements in ``obj_sub`` must be present
            in ``obj_sub`` at the same nesting level and position, but dict
            elements in ``obj_super`` may be omitted in ``obj_sub``.

        name_sub (optional): Name of ``obj_sub`` used in error messages.

        name_super (optional): Name of ``obj_super`` used in error messages.

        idx_list (optional): List index of the current objects (if they are
            elements in a sequence) used in error messages; generally only used
            in recursive calls.

        idx_dict (optional): Dict index of the current objects (if they are
            elements in a dict) used in error messages; generally only used
            in recursive calls

    Notes:
        Elements of ``subdict`` of certain types recieve special treatment:
            ``IgnoredElement``: Not not compared to the respective ``superdict``
                elements.

            ``UnequalElement``: Must be unequal to the respective ``superdict``
                elements.

    """
    if ignored(obj_sub) or ignored(obj_super):
        return

    if obj_sub == obj_super:
        return

    # Collect objects passed to exceptions
    err_objs = {
        "name_super": name_super,
        "obj_super": obj_super,
        "name_sub": name_sub,
        "obj_sub": obj_sub,
        "idx_list": idx_list,
        "idx_dict": idx_dict,
    }

    def exception(msg: str, info: Dict[str, Any]) -> Exception:
        """Build exception that can be raised directly."""
        blocks = [msg]
        for name, obj in info.items():
            s_obj = pformat(obj)
            if "\n" in s_obj:
                block = f"{name}:\n{s_obj}"
            else:
                block = f"{name}: {s_obj}"
            blocks.append(block)
        err = "\n\n".join(blocks)
        return CheckFailedError(err)

    # Check types
    t_super, t_sub = type(obj_super), type(obj_sub)
    if not isinstance(obj_sub, t_super) and not isinstance(obj_super, t_sub):
        if isinstance(obj_sub, Sequence) and isinstance(obj_super, Sequence):
            pass
        else:
            raise exception(
                f"incompatible types {t_super.__name__} and {t_sub.__name__} "
                f"(neither is an instance of the other)",
                {**err_objs, "t_super": t_super, "t_sub": t_sub},
            )

    if isinstance(obj_sub, dict):
        for idx_dict_i, (key, val_sub) in enumerate(obj_sub.items()):
            val_super = get_dict_element(obj_super, key, "superdict", CheckFailedError)
            check_is_sub_element(
                val_sub, val_super, name_sub, name_super, idx_list, idx_dict_i
            )

    elif isiterable(obj_sub, str_ok=False):
        # Compare other (non-str) iterables

        if not isiterable(obj_super, str_ok=False):
            raise exception("superdict element not iterable", err_objs)

        n_sub, n_super = len(obj_sub), len(obj_super)
        if n_sub != n_super:
            raise exception(
                f"iterable elements differ in size: {n_sub} != {n_super}",
                {**err_objs, "n_super": n_super, "n_sub": n_sub},
            )

        for idx, (subobj_sub, subobj_super) in enumerate(zip(obj_sub, obj_super)):
            try:
                check_is_sub_element(
                    subobj_sub,
                    subobj_super,
                    name_sub,
                    name_super,
                    idx_list=idx,
                    idx_dict=idx_dict,
                )
            except CheckFailedError as error:
                msg = str(error).split("\n", 1)[0]
                raise exception(
                    f"iterable elements #{idx} differ: {msg}",
                    {
                        **err_objs,
                        "sub_obj_super": subobj_super,
                        "sub_obj_sub": subobj_sub,
                    },
                ) from error

    else:
        raise exception(f"elements differ ('{name_sub}' vs. '{name_super}')", err_objs)


def get_dict_element(dict_, key, name="dict", exception_type=ValueError):
    """Get an element from a dict, raising an exception otherwise."""
    try:
        return dict_[key]
    except KeyError:
        err = f"key missing in {name}: {key}"
    except TypeError:
        err = f"{name} has wrong type: {type(dict_)}"
    lines = pformat(dict_).split("\n")
    if len(lines) > 10:
        lines = lines[:5] + ["..."] + lines[-5:]
    raise exception_type(err, {"name": name, "key": key, "dict_": dict_})


@dc.dataclass(frozen=True)
class TestConfBase:
    def derive(self, **kwargs):
        # pylint: disable=E1101  # no-member (__dataclass_fields__)
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update(kwargs)
        return type(self)(**data)


def check_is_list_like(obj, *args, **kwargs):
    is_list_like(obj, *args, raise_=CheckFailedError, **kwargs)


def return_or_raise(msg, kwargs, raise_):
    if not raise_:
        return False
    if kwargs is None:
        kwargs = {}
    raise Exception(msg, kwargs)


def is_list_like(
    obj, *, len_=None, not_=None, t_children=None, f_children=None, raise_=False
):
    """Assert that an object is list-like, with optional additional checks.

    Args:
        obj (type): Presumably list-like object.

        len_ (int, optional): Length of list-like object. Defaults to None.

        not_ (type or list[type], optional): Type(s) that ``obj`` must not be
            an instance of. Defaults to None.

        t_children (type or list[type], optional): Type(s) that the elements in
            ``obj`` must be an instance of. Defaults to None.

        f_children (callable, optional): Function used in assert with each
            element in ``obj``. Defaults to None.

        raise_ (bool, optional): Raise an exception instead of returning False.
            Defaults to False.

    """
    kwargs = {
        "obj": obj,
        "len_": len_,
        "not_": not_,
        "t_children": t_children,
        "f_children": f_children,
    }

    if not isiterable(obj, str_ok=False):
        return_or_raise(
            f"{type(obj).__name__} instance `obj` is not iterable", kwargs, raise_
        )

    if len_ is not None:
        if len(obj) != len_:
            return_or_raise(
                f"obj has wrong length: {len(obj)} != {len_}", kwargs, raise_
            )

    if not_ is not None:
        if isinstance(obj, not_):
            return_or_raise(
                f"obj has unexpected type {type(obj).__name__}", kwargs, raise_
            )

    _check_children(obj, t_children, f_children, kwargs, raise_)

    return True


def _check_children(obj, t_children, f_children, kwargs, raise_):

    if t_children is not None:
        for idx, child in enumerate(obj):
            if not isinstance(child, t_children):
                return_or_raise(
                    "child has unexpected type",
                    {**kwargs, "child": child, "idx": idx},
                    raise_,
                )

    if f_children is not None:
        for idx, child in enumerate(obj):
            if not f_children(child):
                return_or_raise(
                    "f_children returns False for child",
                    {**kwargs, "child": child, "idx": idx},
                    raise_,
                )


def type_name(type_) -> str:
    try:
        return type_.__name__
    except AttributeError as e:
        if str(type_).startswith("typing."):
            return str(type_).split(".")[1]
        raise ValueError(type_) from e


# pylint: disable=R0915  # too-many-statements
def assert_nested_equal(
    obj1: Collection,
    obj2: Collection,
    name1: Optional[str] = None,
    name2: Optional[str] = None,
    *,
    float_close_ok: bool = False,
    kwargs_close: Optional[Dict[str, Any]] = None,
) -> None:
    """Compare two nested collections (dicts etc.) for equality.

    Args:
        obj1: Object compared against ``obj2``.

        obj2: Object compared against ``obj1``.

        name1 (optional): Descriptive name for ``obj1`` (e.g., "result").

        name2 (optional): Descriptive name for ``obj2`` (e.g., "solution").

        float_close_ok (optional): Whether it is sufficient for floats to be
            close instead of identical.

        kwargs_close (optional): Keyword arguments passed to ``np.close``.

    """
    if not isinstance(obj1, Collection):
        raise ValueError(f"expecting Collection, not {type_name(type(obj1))}")
    if not isinstance(obj2, Collection):
        raise ValueError(f"expecting Collection, not {type_name((obj2))}")

    def format_obj(obj):
        lines = pformat(obj).split("\n")
        if len(lines) > 20:
            lines = lines[:10] + ["..."] + lines[-10:]
        return "\n".join(lines)

    def error(msg, path, obj1=None, obj2=None):
        err = f"\n{msg}\n\nPath ({len(path)}):\n{pformat(path)}\n"
        if obj1 is not None:
            err += f"\n[{name1 or 'obj1'}] ({type_name(type(obj1))}):"
            err += f"\n{format_obj(obj1)}\n"
        if obj2 is not None:
            err += f"\n[{name2 or 'obj2'}] ({type_name(type(obj2))}):"
            err += f"\n{format_obj(obj2)}\n"
        return AssertionError(err)

    # pylint: disable=R0912  # too-many-branches
    def recurse(obj1, obj2, path):
        try:
            if obj1 == obj2:
                return
        except ValueError:
            # Numpy array?
            try:
                arrays_equal = (obj1 == obj2).all()
            except Exception:  # pylint: disable=W0703  # broad-except
                pass
            else:
                if arrays_equal:
                    return

        def check_equivalent(obj1, obj2, type_, path):
            if not isinstance(obj1, type_) or not isinstance(obj2, type_):
                raise error(
                    f"unequivalent types: expected {type_name(type_)}, got "
                    f"{type_name(type(obj1))} and {type_name(type(obj2))}",
                    *[path, obj1, obj2],
                )

        if obj1 is None or obj2 is None:
            raise error("one is None, the other not", path, obj1, obj2)
        elif isinstance(obj1, str):
            check_equivalent(obj1, obj2, str, path)
            # Must be unequal, otherwise already returned above
            raise error("unequal strings", path, obj1, obj2)
        elif isinstance(obj1, Mapping):
            check_equivalent(obj1, obj2, Mapping, path)
            if obj1.keys() != obj2.keys():
                raise error(
                    f"mappings differ in keys: {obj1.keys()} vs. {obj2.keys()}",
                    path,
                    obj1,
                    obj2,
                )
            for key, val1 in obj1.items():
                val2 = obj2[key]
                recurse(val1, val2, path + [f"key: {key}"])

        elif isinstance(obj1, Sequence):
            check_equivalent(obj1, obj2, Sequence, path)
            if len(obj1) != len(obj2):
                # pylint: disable=E1121  # too-many-function-args
                raise error(
                    f"sequences differ in length: {len(obj1)} != {len(obj2)}",
                    path,
                    obj1,
                    obj2,
                )
            for idx, (ele1, ele2) in enumerate(zip(obj1, obj2)):
                recurse(ele1, ele2, path + [f"idx: {idx}"])

        elif isinstance(obj1, Collection):
            check_equivalent(obj1, obj2, Collection, path)
            if len(obj1) != len(obj2):
                # pylint: disable=E1121  # too-many-function-args
                raise error(
                    f"collections differ in length: {len(obj1)} != {len(obj2)}",
                    path,
                    obj1,
                    obj2,
                )
            try:
                obj1 = sorted(obj1)
                obj2 = sorted(obj2)
            except Exception as e:
                raise error(
                    "unequal collections are unsortable", path, obj1, obj2
                ) from e
            for idx, (ele1, ele2) in enumerate(zip(obj1, obj2)):
                recurse(ele1, ele2, path + [f"idx: {idx}"])

        elif np.isreal(obj1):
            if not np.isreal(obj2):
                raise error(
                    f"unequivalent types (expected real numbers): "
                    f"{type_name(type(obj1))} vs. {type_name(type(obj2))}, ",
                    path,
                    obj1,
                    obj2,
                )
            if np.isnan(obj1) and np.isnan(obj2):
                return
            elif (
                isinstance(obj1, float) or isinstance(obj2, float)
            ) and float_close_ok:
                if np.isclose(obj1, obj2, **(kwargs_close or {})):
                    return
                msg = f"unequal floats not even close: {obj1} vs. {obj2}"
                if kwargs_close:
                    msg += " ({})".format(
                        ", ".join([f"{k}={sfmt(v)}" for k, v in kwargs_close.items()]),
                    )
                raise error(msg, path)
            if isinstance(obj1, bool) and isinstance(obj2, bool):
                msg = f"unequal bools: {obj1} vs. {obj2}"
            elif isinstance(obj1, int) and isinstance(obj2, int):
                msg = f"unequal ints: {obj1} vs. {obj2}"
            else:
                msg = f"unequal reals: {obj1} vs. {obj2} (consider float_close_ok)"
            raise error(msg, path)
        else:
            raise error("unequal objects", path, obj1, obj2)

    return recurse(obj1, obj2, path=[])
