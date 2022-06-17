"""Ensure interoperability by extracting power, energy, price, revenue, and
dimensionless values/timeseries from data."""

from ... import testing
from ...tools import nits, stamps, frames
from pint import DimensionalityError
from typing import Dict, Mapping, Union, Iterable
from dataclasses import dataclass
import pandas as pd

_ATTRIBUTES = ("w", "q", "p", "r", "nodim", "agn")


@dataclass
class InOp:
    """Class to check increase interoperability. Tries to extract power (w), energy (q),
    price (p), revenue (r), adimensional (nodim) and dim-agnostic (agn) information from
    the provided data; initially without checking for consistency."""

    w: Union[nits.Q_, pd.Series] = None
    q: Union[nits.Q_, pd.Series] = None
    p: Union[nits.Q_, pd.Series] = None
    r: Union[nits.Q_, pd.Series] = None
    nodim: Union[nits.Q_, pd.Series] = None  # explicitly dimensionless
    agn: Union[float, pd.Series] = None  # agnostic

    def __post_init__(self):
        # Add correct units and check type.
        self.w = _prepvalue(self.w, "MW")
        self.q = _prepvalue(self.q, "MWh")
        self.p = _prepvalue(self.p, "Eur/MWh")
        self.r = _prepvalue(self.r, "Eur")
        self.nodim = _prepvalue(self.nodim, "")
        self.agn = _prepvalue(self.agn, None)

    @classmethod
    def from_data(cls, data):
        return _from_data(data)

    def to_timeseries(self, index=None):
        """Turn all values into timeseries or None. If none of the attributes is a
        timeseries, and no ``index`` is provided, raise Error. If >1 is a timeseries,
        store only the timestamps where they overlap (i.e., intersection)."""
        # Get index.
        indices = [] if index is None else [index]
        for attr in _ATTRIBUTES:
            if isinstance(val := getattr(self, attr), pd.Series):
                indices.append(val.index)
        index = stamps.intersection(*indices)  # raises error if none passed
        if not len(index):
            raise ValueError("Data has no overlapping timestamps.")
        # Save all values as timeseries.
        kwargs = {}
        for attr in _ATTRIBUTES:
            val = getattr(self, attr)
            if val is None:
                continue
            elif isinstance(val, pd.Series):
                kwargs[attr] = val.loc[index]
            elif isinstance(val, nits.Q_):
                kwargs[attr] = pd.Series(val.m, index, dtype=nits.pintunit(val.units))
            else:  # float
                kwargs[attr] = pd.Series(val, index)
        return InOp(**kwargs)

    def assign_agn(self, da: str = None):
        """Set dimension-agnostic part as specific dimension (unless it's None)."""
        if self.agn is None or da is None:
            return self
        return self.drop("agn") | InOp(**{da: self.agn})

    def drop(self, da: str):
        """Drop part of the information and return new InOp object."""
        return InOp(**{attr: getattr(self, attr) for attr in _ATTRIBUTES if attr != da})

    def __bool__(self):
        return not all(getattr(self, attr) is None for attr in _ATTRIBUTES)

    def __or__(self, other):
        return _union(self, other)

    __ror__ = __or__

    def __eq__(self, other):
        return _equal(self, other)


def _prepvalue(v, unit):
    """Add unit (if no unit set yet) or convert to unit."""
    if v is None:
        return None

    if unit is None:  # unit-agnostic
        if isinstance(v, int) or isinstance(v, float):
            return float(v)
        if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
            v = _prep_timeseries(v)  # float or pint series
            if hasattr(v, "pint"):
                raise DimensionalityError(
                    "Agnostic timeseries should not have a dimension and should not be "
                    f"dimensionless. Should be plain number values; found {v.pint.units}."
                )
            _assert_standardized(v)
            return v.astype(float)
        raise TypeError(
            f"Value should be a number or timeseries of numbers; got {type(v)}."
        )

    else:  # unit-aware
        if isinstance(v, float) or isinstance(v, int) or isinstance(v, nits.Q_):
            return nits.Q_(v, unit)  # add unit or convert to unit
        if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
            v = _prep_timeseries(v)  # float or pint series
            _assert_standardized(v)
            return v.astype(nits.pintunit(unit))
        raise TypeError(
            f"Value should be a number, Quantity, or timeseries; got {type(v)}."
        )


def _assert_standardized(s: pd.Series):
    try:
        frames.assert_standardized(s)
    except AssertionError as e:
        raise ValueError(
            "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
        ) from e


def _prep_timeseries(s: pd.Series) -> pd.Series:
    """Check if a timeseries is a series of objects, and if so, see if these objects are
    actually Quantities."""

    if s.dtype != object:  # float, int, or pint
        return s if hasattr(s, "pint") else s.astype(float)  # float or pint

    # object -> maybe series of Quantitis -> convert to pint-series.
    if not all(isinstance(val, nits.Q_) for val in s.values):
        raise TypeError(f"Timeseries with unexpected data type: {s.dtype}.")
    quantities = [val.to_base_units() for val in s.values]
    magnitudes = [q.m for q in quantities]
    units = list(set([q.u for q in quantities]))
    if len(units) != 1:
        raise DimensionalityError(
            f"Timeseries with inconsistent dimension; found {','.join(units)}."
        )
    return pd.Series(magnitudes, s.index, dtype=nits.pintunit(units[0]))


def _unit2attr(unit) -> str:
    attr = nits.unit2name(unit)  # Error if dimension unknown
    if attr not in _ATTRIBUTES:
        raise NotImplementedError(f"Cannot handle data with this unit ({unit}).")
    return attr


def _from_data(
    data: Union[float, nits.Q_, pd.Series, Dict, pd.DataFrame, Mapping]
) -> InOp:
    """Turn ``data`` into a InterOp object."""
    if isinstance(data, int) or isinstance(data, float):
        return InOp(agn=data)

    elif isinstance(data, nits.Q_):
        return InOp(**{_unit2attr(data.units): data})

    elif isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        # timeseries
        if hasattr(data, "pint"):  # pint timeseries
            return InOp(**{_unit2attr(data.pint.units): data})
        elif data.dtype == object:  # timeeries of objects -> maybe Quantities?
            if len(data) and isinstance(val := data.values[0], nits.Q_):
                # use unit of first value to find dimension
                return InOp(**{_unit2attr(val.u): data})
        else:  # assume float or int
            return InOp(agn=data)

    elif (
        isinstance(data, pd.DataFrame)
        or isinstance(data, pd.Series)
        or isinstance(data, Mapping)
    ):

        def dimabbr(key):
            if key in _ATTRIBUTES:
                return key
            elif not isinstance(key, str) and isinstance(key, Iterable):
                if (da := dimabbr(key[0])) is not None:
                    return da
                if (da := dimabbr(key[-1])) is not None:
                    return da
            return None

        ios = None
        for key, value in data.items():
            if da := dimabbr(key):
                ios |= InOp(**{da: value})
            else:
                raise KeyError(
                    f"Found item with unexpected key/name '{key}'. Should be one of {', '.join(_ATTRIBUTES)}."
                )
        if ios is None:
            return InOp()  # raises Error
        return ios

    raise TypeError(
        f"Expecting number, Quantity, timeseries, or Mapping (e.g. dict or DataFrame); got {type(data)}."
    )


def _union(interop1: InOp, interop2: InOp) -> InOp:
    """Combine 2 ``InterOp`` objects, and raise error if same dimension is supplied twice."""
    if interop2 is None:
        return interop1
    if not isinstance(interop2, InOp):
        raise TypeError("Can only unite same object type.")
    kwargs = {}
    for attr in _ATTRIBUTES:
        val1, val2 = getattr(interop1, attr), getattr(interop2, attr)
        if val1 is not None and val2 is not None:
            raise ValueError(f"Got two values for attribute '{attr}'.")
        elif val1 is not None:
            kwargs[attr] = val1
        else:
            kwargs[attr] = val2
    return InOp(**kwargs)


def _equal(interop1: InOp, interop2: InOp) -> InOp:
    """2 ``InterOp`` objects are equal if they have the same attributes."""
    if not isinstance(interop2, InOp):
        return False
    for attr in _ATTRIBUTES:
        val1, val2 = getattr(interop1, attr), getattr(interop2, attr)
        if type(val1) is not type(val2):
            return False
        if isinstance(val1, pd.Series):
            try:
                testing.assert_series_equal(val1, val2, check_names=False)
            except AssertionError:
                return False
        elif val1 != val2:
            return False
    return True