"""Data structures."""
# Standard library
from typing import Callable
from typing import Sequence
from typing import Union

# Third-party
import numpy as np


def ensemble_probability(
    arr: np.ndarray, thr: float = 0.0, thr_type: str = "lower"
) -> np.ndarray:
    """Ensemble-based probability of threshold exceedence at each point.

    Args:
        arr: Data array with ensemble members as the first dimension.

        thr (optional): Threshold value for data selection in each member.

        thr_type (optional): Threshold type (lower or upper).

    Returns:
        Field with the number of members with a cloud at each grid point.

    """
    if thr_type == "lower":
        mask = arr > thr
    elif thr_type == "upper":
        mask = arr < thr
    else:
        raise ValueError(
            f"invalid threshold type '{thr_type}' (neither 'lower' nor 'upper')"
        )
    n_mem = arr.shape[0]
    arr = np.count_nonzero(mask, axis=0).astype(np.float32) * 100 / n_mem
    return arr


class Cloud:
    """Particle cloud."""

    def __init__(
        self,
        mask: np.ndarray,
        ts: float = 1.0,
    ) -> None:
        """Create an instance of ``Cloud``.

        Args:
            mask: Cloud mask array with two or more dimensions (time plus one or
                more spatial dimensions).

            ts: Time step duration.

            thr (optional): Threshold value defining a cloud.

        """
        self.mask = np.asarray(mask, np.bool)
        self.ts = ts
        if len(self.mask.shape) < 2:
            raise ValueError(f"mask must be 2D or more, not {len(self.mask.shape)}D")

    def departure_time(self) -> np.ndarray:
        """Time until the last cloud has departed.

        Returns:
            Array with the same shape as ``mask`` containing:

                - -inf: Cloud-free until the end, regardless of what was before.
                - > 0: Time until the last cloud will have departed.
                - inf: A cloud is still present at the last time step.

        """
        arr = np.full(self.mask.shape, -np.inf, np.float32)

        # Set points with a cloud at the last time step to +INF at all steps
        arr[:, self.mask[-1]] = np.inf

        # Points without a cloud until the last time step
        m_clear_till_end = self.mask[::-1].cumsum(axis=0)[::-1] == 0

        # Points where the last cloud disappears at the next time step
        m_last_cloud = np.concatenate(
            [
                (m_clear_till_end[1:].astype(int) - m_clear_till_end[:-1]),
                np.zeros([1] + list(self.mask.shape[1:])),
            ],
            axis=0,
        ).astype(bool)

        # Points where a cloud will disappear before the last time step
        m_will_disappear = m_last_cloud[::-1].cumsum(axis=0)[::-1].astype(np.bool)

        # Set points where a cloud will disappear to the time until it's gone
        arr[:] = np.where(
            m_will_disappear,
            m_will_disappear[::-1].cumsum(axis=0)[::-1] * self.ts,
            arr,
        )

        return arr

    def arrival_time(self) -> np.ndarray:
        """Time until the first cloud has arrived.

        Returns:
            Array with the same shape as ``mask`` containing:

                - inf: Cloud-free until the end, regardless of what was before.
                - > 0: Time until the first cloud will have arrived.
                - < 0: Time since the before first cloud has arrived.
                - -inf: A cloud has been present since the first time step.

        """
        arr = np.full(self.mask.shape, np.inf, np.float32)

        # Points without a cloud since the first time step
        m_clear_since_start = self.mask.cumsum(axis=0) == 0

        # Points without a cloud until the last time step
        m_clear_till_end = self.mask[::-1].cumsum(axis=0)[::-1] == 0

        # Set points that have been cloudy since the start to -INF
        arr[self.mask[:1] & ~m_clear_till_end] = -np.inf

        # Points where the first cloud has appeard during the previous time step
        m_first_cloud = np.concatenate(
            [
                (m_clear_since_start[1:].astype(int) - m_clear_since_start[:-1]),
                np.zeros([1] + list(self.mask.shape[1:])),
            ],
            axis=0,
        ).astype(bool)

        # Points where first cloud will appear before the last time step
        m_will_appear = m_first_cloud[::-1].cumsum(axis=0)[::-1].astype(np.bool)

        # Set points where first cloud will appear to the time until it's there
        arr[:] = np.where(
            m_will_appear, m_will_appear[::-1].cumsum(axis=0)[::-1] * self.ts, arr
        )

        # Points where first cloud has appeared before the current time step
        m_has_appeared = (
            ~m_clear_since_start & ~m_will_appear & ~m_clear_till_end & ~self.mask[:1]
        )

        # Set points where first cloud has appeared to time since before it has
        arr[:] = np.where(m_has_appeared, -m_has_appeared.cumsum(axis=0) * self.ts, arr)

        return arr


# SR_TODO Eliminate EnsembleCloud once Cloud works
class EnsembleCloud(Cloud):
    """Particle cloud in an ensemble simulation."""

    def __init__(self, mask: np.ndarray, mem_min: int = 1, ts: float = 1.0) -> None:
        """Create in instance of ``EnsembleCloud``.

        Args:
            mask: Cloud mask array with at least three dimensions (ensemble
                members, time and one or more spatial dimensions).

            mem_min: Minimum number of members required per grid point to define
                the ensemble cloud.

            ts (optional): Time step duration.

        """
        mask = np.asarray(mask, np.bool)
        if len(mask.shape) < 3:
            raise ValueError(f"mask must be 3D or more, not {len(mask.shape)}D")
        mask = np.count_nonzero(mask, axis=0) >= mem_min
        super().__init__(mask=mask, ts=ts)


def merge_fields(
    flds: Sequence[np.ndarray], op: Union[Callable, Sequence[Callable]] = np.nansum
) -> np.ndarray:
    """Merge fields by applying a single operator or an operator chain.

    Args:
        flds: Fields to be merged.

        op (optional): Opterator(s) used to combine input fields. Must accept
            argument ``axis=0`` to only reduce along over the fields.

            If a single operator is passed, it is used to sequentially combine
            one field after the other, in the same order as the corresponding
            specifications (``var_setups``).

            If a list of operators has been passed, then it's length must be
            one smaller than that of ``var_setups``, such that each
            operator is used between two subsequent fields (again in the same
            order as the corresponding specifications).

    """
    if callable(op):
        return op(flds, axis=0)
    elif isinstance(op, Sequence):
        op_lst = op
        if not len(flds) == len(op_lst) + 1:
            raise ValueError("wrong number of fields", len(flds), len(op_lst) + 1)
        fld = flds[0]
        for i, fld_i in enumerate(flds[1:]):
            _op = op_lst[i]
            fld = _op([fld, fld_i], axis=0)
        return fld
    else:
        raise Exception("no operator(s) defined")
