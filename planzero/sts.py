import array
import bisect
import builtins
from enum import Enum
import functools

import numpy as np
import pint
from pydantic import BaseModel

from .ureg import ureg as u
from . import objtensor


_seconds_per_year = (1 * u.year).to(u.second).magnitude

class InterpolationMode(str, Enum):
    # queries with exact matches will return the corresponding value
    # queries without exact matches will return NaN or raise an exception
    no_interpolation = 'no_interpolation'

    # queries with exact matches will return the previous value
    # queries without exact matches will return the previous value
    latest = 'latest'

    # queries with exact matches will return the corresponding value
    # queries without exact matches will return the previous value
    current = 'current'


class STS(BaseModel):
    """A data structure of (time, value) pairs (stored separately) representing
    a timeseries. It may or not have a default value.
    It is unit-aware.
    It may be defined by interpolation for all time, some time, or only
    specific times.
    """

    t_unit:object
    v_unit:object

    times:object # will be a double-precision float array
    values:object # will be a double-precision float array

    current_readers:list[str] = []
    writer:str|None = None

    identifier: str | None = None

    interpolation: InterpolationMode

    @classmethod
    def zero_one(cls, time, interpolation=InterpolationMode.current, v_unit=None):
        rval = cls(
            times=array.array('d', [time.magnitude]),
            t_unit=time.u,
            values=array.array('d', [0, 1]),
            v_unit=v_unit or u.dimensionless,
            interpolation=InterpolationMode.current)
        return rval

    @classmethod
    def one_zero(cls, time, interpolation=InterpolationMode.current, v_unit=None):
        rval = cls(
            times=array.array('d', [time.magnitude]),
            t_unit=time.u,
            values=array.array('d', [1, 0]),
            v_unit=v_unit or u.dimensionless,
            interpolation=InterpolationMode.current)
        return rval

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #assert len(self.times) + 1 == len(self.values), (len(self.times), len(self.values))
        #assert isinstance(self.times, array.array)
        #assert isinstance(self.values, array.array)

    def times_with_units(self):
        return [tt * self.t_unit for tt in self.times]


    def max(self, _i_start=None):
        # TODO: what is this _i_start business??
        if _i_start is None:
            if self.interpolation == InterpolationMode.no_interpolation:
                rval = max(self.values[1:])
            else:
                rval = max(self.values)
        else:
            rval = max(self.values[_i_start:])
        return rval * self.v_unit

    def __string__(self):
        return f'STS(id={self.identifier})'

    def _init_v_unit(self, values, unit, default_value):
        # specify unit if you want no default value and you don't know any values yet
        # provide default_value if you do want a default value to apply prior to the first (time, value) pair, then no unit
        if default_value is None:
            # without a default_unit, it's required to either
            # (a) set the unit, or
            # (b) provide initial values and times
            # and it is okay to do both, but then `unit` takes precedence.
            if unit is None:
                return values[0].u
            else:
                if isinstance(unit, str):
                    unit = getattr(u, unit)
                return unit
        else:
            return default_value.u



    def __len__(self):
        return len(self.times)

    def _idx_of_time(self, t_query):
        """Return the index into the `values` array corresponding to time t_query.

        N.B. that this function can return different values after appending or
        extending the timeseries.
        """
        ts = t_query.to(self.t_unit).magnitude
        if self.times:
            if ts > self.times[-1]:
                if self.interpolation == InterpolationMode.no_interpolation:
                    index = 0
                    valid = False
                else:
                    index = len(self.values) - 1
                    valid = True
            elif ts == self.times[-1]:
                if self.interpolation == InterpolationMode.no_interpolation:
                    index = len(self.values) - 1
                elif self.interpolation == InterpolationMode.current:
                    index = len(self.values) - 1
                elif self.interpolation == InterpolationMode.latest:
                    index = len(self.values) - 2
                valid = True
            else:
                if self.interpolation == InterpolationMode.no_interpolation:
                    index = bisect.bisect_right(self.times, ts)
                    valid = self.times[index - 1] == ts
                elif self.interpolation == InterpolationMode.current:
                    index = bisect.bisect_right(self.times, ts)
                    valid = True
                elif self.interpolation == InterpolationMode.latest:
                    index = bisect.bisect_left(self.times, ts)
                    valid = True
                else:
                    raise NotImplementedError(self.interpolation)
        else:
            index = 0
            valid = (self.interpolation != InterpolationMode.no_interpolation)
        return index, valid

    def query(self, t_query):
        try:
            n_queries = len(t_query)
        except:
            n_queries = 1
        if n_queries > 1:
            idxs, valids = zip(*[self._idx_of_time(tqi) for tqi in t_query])
            values = [self.values[idx] for idx in idxs]
            rval = np.asarray(values)
            rval[~np.asarray(valids)] = float('nan')
            return rval * self.v_unit
        else:
            idx, valid = self._idx_of_time(t_query)
            if valid:
                return self.values[idx] * self.v_unit
            else:
                assert valid
                return float('nan') * self.v_unit

    def append(self, t, v):
        if t.u == self.t_unit:
            tt = t.magnitude
        elif t.u == u.year and self.t_unit == u.second:
            tt = t.magnitude * _seconds_per_year
        else:
            tt = t.to(self.t_unit).magnitude
        if len(self.times):
            assert tt > self.times[-1]
        self.times.append(tt)
        self.values.append(v.to(self.v_unit).magnitude)

    def extend(self, times, values, skip_nan_values=False):
        assert len(times) == len(values)
        for t, v in zip(times, values):
            if skip_nan_values and np.isnan(v.magnitude):
                continue
            self.append(t, v)

    def plot(self, t_unit=None, annotate=True, **kwargs):
        import matplotlib.pyplot as plt
        # also re: interpolation
        if isinstance(t_unit, str):
            t_unit = getattr(u, t_unit)
        elif t_unit is None:
            t_unit = self.t_unit
        if t_unit != self.t_unit:
            t_scalar = (1.0 * self.t_unit / t_unit).to('dimensionless').magnitude
            times = [tt * t_scalar for tt in self.times]
        else:
            times = self.times
        plt.scatter(
            self.times,
            self.values[1:],
            **kwargs)
        plt.xlabel(t_unit)
        plt.ylabel(self.v_unit)
        plt.title(self.identifier)
        if annotate:
            self.annotate_plot(t_unit=t_unit, **kwargs)

    def to(self, v_unit):
        if isinstance(v_unit, str):
            v_unit = getattr(u, v_unit)
        scalar = (1.0 * self.v_unit / v_unit).to('dimensionless').magnitude
        default_value = (
            None
            if self.interpolation == InterpolationMode.no_interpolation
            else self.values[0] * scalar * v_unit)
        rval = SparseTimeSeries(
            times=[tt * self.t_unit for tt in self.times],
            values=[v * scalar * v_unit for v in self.values[1:]],
            default_value=default_value,
            t_unit=self.t_unit,
            unit=v_unit,
            identifier=None,
            interpolation=self.interpolation)
        return rval

    def annotate_plot(self, t_unit=None, **kwargs):
        """Called once per variable name in comparison plots"""
        pass

    def __add__(self, other):
        if isinstance(other, pint.Quantity):
            if other.magnitude == 0:
                # TODO: check units
                return self * 1
            raise NotImplementedError()
        self_nointerp = (self.interpolation == InterpolationMode.no_interpolation)
        other_nointerp = (other.interpolation == InterpolationMode.no_interpolation)

        if self_nointerp and other_nointerp:
            return add_nointerp_nointerp(self, other)
        elif self_nointerp and not other_nointerp:
            return add_nointerp_interp(self, other)
        elif not self_nointerp and other_nointerp:
            return add_nointerp_interp(other, self)
        elif all(vv == 0 for vv in other.values):
            return self.copy()
        else:
            raise NotImplementedError()

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return scale(self, -1)

    def __sub__(self, other):
            return (self + (-other))

    def __rsub__(self, other):
        return (other + (-self))

    def __truediv__(self, other):
        default_value = (
            None
            if self.interpolation == InterpolationMode.no_interpolation
            else self.values[0] * self.v_unit / other)
        rval = SparseTimeSeries(
            times=[tt * self.t_unit for tt in self.times],
            values=[v / other * self.v_unit for v in self.values[1:]],
            default_value=default_value,
            t_unit=self.t_unit,
            unit=(self.v_unit / other).u,
            identifier=None,
            interpolation=self.interpolation)
        return rval

    def __rtruediv__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        if isinstance(other, STS):
            return mul_sts_sts(self, other)
        elif isinstance(other, (int, float)):
            return scale(self, other)
        elif isinstance(other, pint.Quantity) and isinstance(other.magnitude, (int, float)):
            return scale_convert(self, other)
        return NotImplemented
        #raise NotImplementedError(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def copy(self):
        return self * 1

    def sum(self):
        assert self.interpolation == InterpolationMode.no_interpolation
        rval = sum(self.values[1:]) * self.v_unit
        return rval

    def integral(self, start_time=None, end_time=None):
        raise NotImplementedError()

    def _setdefault_scalar(self, times, val):
        assert self.interpolation == InterpolationMode.no_interpolation
        as_d = dict(zip(self.times, self.values[1:]))
        for ttu in times:
            tt = ttu.to(self.t_unit).magnitude
            as_d.setdefault(tt, val)
        times, values = zip(*list(sorted(as_d.items())))
        self.times = array.array('d', times)
        self.values = array.array('d', self.values[:1])
        self.values.extend(values)

    def setdefault_zero(self, times):
        # Consider also setdefault_one and setdefault_nan
        # I feel like setdefault(...) should require units on values, and
        # maybe it should be a dictionary / vectorized update?
        # it is convenient to not require units on 0 and 1
        # so that templatized code can prepare arguments for arithmetic identities
        # without bothering about units
        # e.g.
        # years_i_care_about = foo()
        # a.setdefault_zero(years) + b.setdefault_zero(years)
        # ...
        # will produce a result that is at least defined for the given years
        return self._setdefault_scalar(times, 0)


class SparseTimeSeries(STS):

    def __init__(self, *, times=None, values=None, unit=None, identifier=None, t_unit=u.seconds, default_value=None,
                 interpolation='current', skip_nan_values=False):
        super().__init__(
            t_unit=t_unit,
            v_unit=self._init_v_unit(values, unit, default_value),
            times=array.array('d'),
            values=array.array('d'),
            current_readers=[],
            writer=None,
            identifier=identifier,
            interpolation=interpolation)

        if self.v_unit == u.kg_CO2 ** 2:
            assert 0

        if interpolation == InterpolationMode.no_interpolation:
            assert default_value is None

        if default_value is None:
            self.values.append(float('nan'))
        else:
            self.values.append(default_value.to(self.v_unit).magnitude)
        if times is not None:
            self.extend(times, values, skip_nan_values=skip_nan_values)


def annual_report(**kwargs):
    return SparseTimeSeries(
        t_unit=u.years,
        interpolation='no_interpolation',
        **kwargs)


def annual_report2(years, values, v_unit, include_nan_values=True):
    # This is a faster implementation of annual_report
    array_values = array.array('d', [float('nan')])
    array_times = array.array('d')
    assert len(years) == len(values)
    if include_nan_values:
        array_times.extend(years)
        array_values.extend(values)
    else:
        for tt, vv in zip(years, values):
            if not np.isnan(vv):
                array_times.append(tt)
                array_values.append(vv)
    return STS(
        t_unit=u.years,
        v_unit=v_unit,
        times=array_times,
        values=array_values,
        current_readers=[],
        writer=None,
        identifier=None,
        interpolation=InterpolationMode.no_interpolation)


def annual_report_decay(self, timescale, horizon):
    """An operation to spread annual sums out forward over time according to
    a first order process """
    assert np.isnan(self.values[0])
    assert all(tt == int(tt) for tt in self.times)
    assert self.t_unit == u.years
    assert self.interpolation == InterpolationMode.no_interpolation

    v_by_t = {int(tt): vv
              for tt, vv in zip(self.times, self.values[1:])}
    v_running = 0.0
    coef = 1.0 / timescale.to(u.years).magnitude
    times = range(int(self.times[0]), int(horizon.to('years').magnitude))
    u_times = []
    values = []
    for tt in times:
        v_running += v_by_t.get(tt, 0)
        values.append(v_running * coef * self.v_unit)
        u_times.append(tt * self.t_unit)
        v_running -= coef * v_running

    return SparseTimeSeries(
        t_unit=u.years,
        times=u_times,
        values=values,
        interpolation='no_interpolation')


def add_nointerp_nointerp(self, other):
    assert self.interpolation == InterpolationMode.no_interpolation
    assert other.interpolation == InterpolationMode.no_interpolation

    default_value = None
    interpolation = InterpolationMode.no_interpolation
    if self.t_unit != other.t_unit:
        raise NotImplementedError()
    t_unit = self.t_unit
    other_as_dict = {tt:vv for tt, vv in zip(other.times, other.values[1:])}
    times = []
    self_values = []
    other_values = []
    for tt, vv in zip(self.times, self.values[1:]):
        try:
            other_values.append(other_as_dict[tt])
            times.append(tt)
            self_values.append(vv)
        except KeyError:
            continue
    rval_values = array.array('d', [float('nan')])
    rval_times = array.array('d', times)
    if times:
        typed_self_values = np.asarray(self_values) * self.v_unit
        typed_other_values = np.asarray(other_values) * other.v_unit
        typed_rval_values = typed_self_values + typed_other_values
        rval_values.extend(typed_rval_values.magnitude)
        rval = STS(
            times=rval_times,
            values=rval_values,
            t_unit=t_unit,
            v_unit=typed_rval_values.u,
            identifier=None,
            interpolation=InterpolationMode.no_interpolation)
    else:
        rval = STS(
            times=rval_times,
            values=rval_values,
            t_unit=t_unit,
            v_unit=self.u,
            identifier=None,
            interpolation=InterpolationMode.no_interpolation)
    return rval


def add_nointerp_interp(self, other):
    assert self.interpolation == InterpolationMode.no_interpolation
    assert other.interpolation != InterpolationMode.no_interpolation

    if self.t_unit != other.t_unit:
        raise NotImplementedError()
    t_unit = self.t_unit
    times = [tt * t_unit for tt in self.times]
    other_values_at_times = other.query(times)
    values = [vv * self.v_unit + ov
              for vv, ov in zip(self.values[1:], other_values_at_times)]
    if values:
        v_unit = values[0].u
    else:
        v_unit = (1 * self.v_unit + 1 * other.v_unit).u
    rval = SparseTimeSeries(
        times=times,
        values=values,
        default_value=None,
        t_unit=t_unit,
        unit=v_unit,
        identifier=None,
        interpolation=InterpolationMode.no_interpolation)
    return rval


def union_times(args, ignore_constants=True):
    if not args:
        raise NotImplementedError()
    qs = [arg for arg in args if isinstance(arg, pint.Quantity) or arg is None]
    nonqs = [arg for arg in args if not isinstance(arg, pint.Quantity) and arg is not None]
    if ignore_constants:
        del qs
    else:
        raise TypeError(qs)

    if len(nonqs) == 0:
        return set()
    elif len(nonqs) == 1:
        return set(nonqs[0].times_with_units())

    t_units = set(arg.t_unit for arg in nonqs)
    if len(t_units) != 1:
        raise NotImplementedError(t_units)
    t_unit = nonqs[0].t_unit

    times = set()
    for arg in nonqs:
        times.update(arg.times)
    return {tt * t_unit for tt in times}


def with_default_zero(self, times):
    # times may be ndarray
    if len(times) and isinstance(self, STS):
        rval = self.copy()
        rval.setdefault_zero(times)
        return rval
    else:
        # This passthrough works for both SparseTimeSeries, and Quantities
        assert not isinstance(self, list)
        return self


def mul_no_interp_no_interp(a, b):
    if a.t_unit != b.t_unit:
        raise NotImplementedError()
    t_unit = a.t_unit
    v_unit = a.v_unit * b.v_unit

    # no interpolation means no default value
    assert np.isnan(a.values[0])
    assert np.isnan(b.values[0])

    a_d = dict(zip(a.times, a.values[1:]))
    c_d = dict()
    for b_t, b_v in zip(b.times, b.values[1:]):
        if b_t in a_d:
            c_d[b_t] = b_v * a_d[b_t]

    c_times, c_values = zip(*sorted(c_d.items()))

    rval = SparseTimeSeries(
        times=[tt * t_unit for tt in c_times],
        values=[vv * v_unit for vv in c_values],
        default_value=None,
        t_unit=t_unit,
        unit=v_unit,
        identifier=None,
        interpolation='no_interpolation')
    return rval


def mul_no_interp_sts(a, b):
    if a.t_unit != b.t_unit:
        raise NotImplementedError()
    t_unit = a.t_unit
    v_unit = a.v_unit * b.v_unit

    # no interpolation means no default value
    assert np.isnan(a.values[0])

    c_times = []
    c_values = []
    for a_t, a_v_nounit in zip(a.times, a.values[1:]):
        b_v = b.query(a_t * t_unit) # has b unit
        c_times.append(a_t * t_unit)
        c_values.append(a_v_nounit * b_v * a.v_unit)

    rval = SparseTimeSeries(
        times=c_times,
        values=c_values,
        default_value=None,
        t_unit=t_unit,
        unit=v_unit,
        identifier=None,
        interpolation='no_interpolation')
    return rval


def mul_sts_sts(self, other):
    if self.interpolation == other.interpolation == 'no_interpolation':
        return mul_no_interp_no_interp(self, other)
    elif self.interpolation == 'no_interpolation':
        return mul_no_interp_sts(self, other)
    elif other.interpolation == 'no_interpolation':
        return mul_no_interp_sts(other, self)
    elif all(vv == 0 for vv in other.values):
        return scale(self, 0)
    raise NotImplementedError()


def scale(self, amount):
    assert isinstance(amount, (float, int))
    rval_times = array.array('d', self.times)
    rval_values = array.array('d', [vv * amount for vv in self.values])
    rval = STS(
        times=rval_times,
        values=rval_values,
        t_unit=self.t_unit,
        v_unit=self.v_unit,
        interpolation=self.interpolation)
    return rval


def scale_convert(self, amount):
    v_coef = amount * self.v_unit
    if self.interpolation == InterpolationMode.no_interpolation:
        default_value = None
    else:
        default_value = self.values[0] * v_coef
    rval = SparseTimeSeries(
        times=[tt * self.t_unit for tt in self.times],
        values=[vv * v_coef for vv in self.values[1:]],
        default_value=default_value,
        t_unit=self.t_unit,
        unit=v_coef.u,
        identifier=None,
        interpolation=self.interpolation)
    return rval

if SparseTimeSeries not in objtensor._types_for_pint_to_ignore:
    objtensor._types_for_pint_to_ignore = (
        objtensor._types_for_pint_to_ignore
        + (STS,))


