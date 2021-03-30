"""Summarize objects as a dict for testing etc."""
# Standard library
import dataclasses as dc
from functools import partial
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Optional
from typing import Sequence

# Third-party
import numpy as np
from cartopy.crs import Projection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.transforms import Transform
from matplotlib.transforms import TransformedBbox

# First-party
from srutils.dataclasses import asdict
from srutils.iter import isiterable

# Local
from .exceptions import NotSummarizableError


def is_attrs_class(cls: Any) -> bool:
    """Determine whether a class has been defined with ``@attr.attrs``."""
    return isinstance(cls, type) and hasattr(cls, "__attrs_attrs__")


def summarize(self: Any) -> Dict[str, Any]:
    """Summarize an object."""
    return Summarizer().run(self)


# pylint: disable=R0912  # too-many-branches (>12)
# pylint: disable=W0621  # redefined-outer-name
# Omit return type ('Callable') to avoid hiding decorated classes from discovery
# (Method usages of decorated classes no longer detected by vscode)
def summarizable(
    cls: Optional[Callable] = None,
    *,
    attrs: Optional[Collection[str]] = None,
    attrs_add: Optional[Collection[str]] = None,
    attrs_skip: Optional[Collection[str]] = None,
    summarize: Optional[Callable[[Any], Dict[str, Any]]] = None,
    auto_collect: bool = True,
    overwrite: bool = False,
):
    """Decorate a class to make it summarizable.

    Args:
        cls: Class to be decorated.

        attrs: Class attributes to summarize. Added to ``cls`` as class
            attribute ``summarizable_attrs``, which also contains all auto-
            collected attributes of special classes like dataclasse (unless
            ``auto_collect`` is False), but none specified in ``attrs_skip``.

        attrs_add: Class attributes to summarize in addition to auto-collected
            attributes of special classes like dataclasses.

        attrs_skip: Class attributes not to summarize. Affects attributes in
            ``attrs`` and, more importantly, auto-collected ones such as
            dataclass attributes.

        summarize: Custom function to summarize the class as a dict containing
            the summarized attributes. Replaces``summarize``. Added to ``cls``
            as method ``summarize``.

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
            attrs_add=attrs_add,
            attrs_skip=attrs_skip,
            summarize=summarize,
            auto_collect=auto_collect,
            overwrite=overwrite,
        )

    if attrs is not None:
        attrs = list(attrs)
    else:
        if auto_collect and is_attrs_class(cls):
            # Collect attributes defined with ``attr.attrib``
            attrs = [attr.name for attr in cls.__attrs_attrs__]  # type: ignore
        elif auto_collect and dc.is_dataclass(cls):
            # Collect dataclass fields
            attrs = list(cls.__dataclass_fields__)  # type: ignore
        else:
            attrs = []
    attrs += list(attrs_add or [])
    if attrs_skip:
        attrs = [name for name in attrs if name not in attrs_skip]

    if hasattr(cls, "__summarize__") and "__summarize__" not in cls.__dict__:
        super_summarize = getattr(cls, "__summarize__")
    else:
        super_summarize = None

    def __summarize__(self) -> Dict[str, Any]:
        dct: Dict[str, Any]
        if summarize is not None:
            dct = summarize(self)
        else:
            if super_summarize is None:
                dct = {"type": type(self).__name__}
            else:
                dct = super_summarize(self)
        assert attrs is not None  # mypy
        for name in attrs:
            dct[name] = getattr(self, name)
        return dct

    setattr(cls, "__summarize__", __summarize__)

    return cls


class Summarizer:
    """Summarize an object as a string, list, dict etc."""

    def run(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return obj
        methods = [
            self._try_mpl,
            self._try_cartopy,
            self._try_summarizable,
            self._try_dict_like,
            self._try_list_like,
            self._try_named,
            self._try_eval,
            self._try_eval_np,
        ]
        for method in methods:
            try:
                obj = method(obj)
            except NotSummarizableError:
                continue
            else:
                break
        else:
            return str(obj)
        return obj

    # pylint: disable=R0201  # no-self-use
    def _try_mpl(self, obj: Any) -> Dict[str, Any]:
        """Try to summarize ``obj`` as a matplotlib object."""
        if isinstance(obj, Figure):
            return summarize_mpl_figure(obj)
        elif isinstance(obj, Axes):
            return summarize_mpl_axes(obj)
        elif isinstance(obj, (Bbox, TransformedBbox)):
            return summarize_mpl_bbox(obj)
        elif isinstance(obj, (Bbox, Transform)):
            return summarize_mpl_transform(obj)
        raise NotSummarizableError("mpl", obj)

    # pylint: disable=R0201  # no-self-use
    def _try_cartopy(self, obj: Any) -> Dict[str, Any]:
        """Try to summarize ``obj`` as a cartopy object."""
        if isinstance(obj, Projection):
            return summarize_cartopy_projection(obj)
        raise NotSummarizableError("cartopy", obj)

    def _try_summarizable(self, obj: Any) -> Dict[str, Any]:
        """Try to summarize ``obj`` as a summarizable object."""
        try:
            dct = obj.__summarize__()
        except AttributeError as e:
            raise NotSummarizableError("summarizable", obj) from e
        return self.run(dct)

    def _try_dict_like(self, obj: Any) -> Dict[Any, Any]:
        """Try to summarize ``obj`` as a dict-like object."""
        try:
            items = obj.items()
        except AttributeError:
            try:
                obj = dict(obj)
            except (TypeError, ValueError):
                try:
                    obj = obj.dict()
                except AttributeError as e:
                    try:
                        obj = asdict(obj, shallow=True)
                    except TypeError:
                        raise NotSummarizableError("dict-like", obj) from e
            items = obj.items()
        return {self.run(key): self.run(val) for key, val in items}

    def _try_list_like(self, obj: Any) -> Sequence[Any]:
        """Try to summarize ``obj`` as a list-like object."""
        if not isiterable(obj, str_ok=False):
            raise NotSummarizableError("list-like", obj)
        data = []
        for item in obj:
            data.append(self.run(item))
        return data

    # pylint: disable=R0201  # no-self-use
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

    # pylint: disable=R0201  # no-self-use
    # pylint: disable=W0123  # eval-used
    def _try_eval(self, obj: Any) -> Any:
        try:
            eval(str(obj))
        except (SyntaxError, NameError) as e:
            raise NotSummarizableError("eval", obj) from e
        return obj

    # pylint: disable=R0201  # no-self-use
    # pylint: disable=W0123  # eval-used
    def _try_eval_np(self, obj: Any) -> Any:
        try:
            eval(f"np.{obj}")
        except (SyntaxError, NameError) as e:
            raise NotSummarizableError("eval_np", obj) from e
        return obj


def summarize_mpl_figure(obj: Any) -> Dict[str, Any]:
    """Summarize a matplotlib ``Figure`` instance as a dict."""
    summary = {
        "type": type(obj).__name__,
        # SR_TODO if necessary, add option for shallow summary (avoid loops)
        "axes": [summarize_mpl_axes(a) for a in obj.get_axes()],
        "bbox": summarize_mpl_bbox(obj.bbox),
    }
    return summary


def summarize_mpl_axes(obj: Any) -> Dict[str, Any]:
    """Summarize a matplotlib ``Axes`` instance as a dict."""
    # Note: Hand-picked selection of attributes (not very systematically)
    summary = {
        "type": type(obj).__name__,
        "adjustable": obj.get_adjustable(),
        "aspect": obj.get_aspect(),
        "bbox": summarize_mpl_bbox(obj.bbox),
        "data_ratio": obj.get_data_ratio(),
        "facecolor": obj.get_facecolor(),
        "fc": obj.get_fc(),
        "frame_on": obj.get_frame_on(),
        "label": obj.get_label(),
        "lines": [np.asarray(line.get_data()).tolist() for line in obj.get_lines()],
        "position": summarize_mpl_bbox(obj.get_position()),
        "title": obj.get_title(),
        "transAxes": summarize_mpl_transform(obj.transAxes),
        "transData": summarize_mpl_transform(obj.transData),
        "visible": obj.get_visible(),
        "window_extent": summarize_mpl_bbox(obj.get_window_extent()),
        "xlabel": obj.get_xlabel(),
        "xlim": obj.get_xlim(),
        "xmajorticklabels": list(map(str, obj.get_xmajorticklabels())),
        "xminorticklabels": list(map(str, obj.get_xminorticklabels())),
        "xscale": obj.get_xscale(),
        "xticklabels": list(map(str, obj.get_xticklabels())),
        "xticks": list(obj.get_xticks()),
        "ylabel": obj.get_ylabel(),
        "ylim": obj.get_ylim(),
        "ymajorticklabels": list(map(str, obj.get_ymajorticklabels())),
        "yminorticklabels": list(map(str, obj.get_yminorticklabels())),
        "yscale": obj.get_yscale(),
        "yticklabels": list(map(str, obj.get_yticklabels())),
        "yticks": list(obj.get_yticks()),
        "zorder": obj.get_zorder(),
    }
    return summary


def summarize_mpl_bbox(obj: Any) -> Dict[str, Any]:
    """Summarize a matplotlib ``Bbox`` instance as a dict."""
    summary = {
        "type": type(obj).__name__,
        "bounds": obj.bounds,
    }
    return summary


def summarize_mpl_transform(obj: Any) -> Dict[str, Any]:
    """Summarize a matplotlib ``Transform`` instance as a dict."""
    summary = {
        "type": type(obj).__name__,
    }
    return summary


def summarize_cartopy_projection(obj: Any) -> Dict[str, Any]:
    """Summarize cartopy ``Projection`` objects."""
    summary = {
        "type": type(obj).__name__,
        "x_limits": obj.x_limits,
        "y_limits": obj.y_limits,
        "proj4_params": obj.proj4_params,
    }
    return summary
