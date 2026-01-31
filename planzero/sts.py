import array
import bisect
import builtins
from enum import Enum

import numpy as np
import pint
from pydantic import BaseModel

from .ureg import ureg as u


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


class SparseTimeSeries(BaseModel):
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

    current_readers:list[str]
    writer:str|None

    identifier: str | None # shouldn't be None after construction

    interpolation: InterpolationMode


    def max(self, _i_start=None):
        if _i_start is None:
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

        if interpolation == InterpolationMode.no_interpolation:
            assert default_value is None

        if default_value is None:
            self.values.append(float('nan'))
        else:
            self.values.append(default_value.to(self.v_unit).magnitude)
        if times is not None:
            self.extend(times, values, skip_nan_values=skip_nan_values)

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
            assert all(valids)

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
            else self.values[0] * scalar * self.v_unit)
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
        if self.interpolation != InterpolationMode.no_interpolation:
            raise NotImplementedError()
        if other.interpolation != InterpolationMode.no_interpolation:
            raise NotImplementedError()
        default_value = None
        interpolation = InterpolationMode.no_interpolation
        if self.t_unit != other.t_unit:
            raise NotImplementedError()
        t_unit = self.t_unit
        other_as_dict = {tt:vv for tt, vv in zip(other.times, other.values[1:])}
        times = []
        values = []
        for tt, vv in zip(self.times, self.values[1:]):
            if tt in other_as_dict:
                times.append(tt * t_unit)
                values.append(vv * self.v_unit + other_as_dict[tt] * other.v_unit)
        if values:
            v_unit = values[0].u
        else:
            v_unit = self.v_unit
        rval = SparseTimeSeries(
            times=times,
            values=values,
            default_value=default_value,
            t_unit=t_unit,
            unit=v_unit,
            identifier=None,
            interpolation=InterpolationMode.no_interpolation)
        return rval

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
        if isinstance(other, SparseTimeSeries):
            return mul_sts_sts(self, other)
        elif isinstance(other, (int, float)):
            return scale(self, other)
        elif isinstance(other, pint.Quantity) and isinstance(other.magnitude, (int, float)):
            return scale_convert(self, other)

        raise NotImplementedError()

def annual_report(**kwargs):
    return SparseTimeSeries(
        t_unit=u.years,
        interpolation='no_interpolation',
        **kwargs)

def mul_sts_sts_no_interp(a, b):
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

def mul_sts_sts(self, other):
    if self.interpolation == other.interpolation == 'no_interpolation':
        return mul_sts_sts_no_interp(self, other)
    raise NotImplementedError()


def usum(args):
    """Sum a set of no_interpolation STSs, but assuming 0 instead of NaN where
    they're undefined, so that it's kind of a "union-sum": the union of times,
    and the sum of values at those times.

    None is a valid argument, it's a signal that isn't defined anywhere, and
    will be ignored.
    """
    args = [arg for arg in args if arg is not None]
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    assert None not in args
    interpolations = set(arg.interpolation for arg in args)
    if len(interpolations) > 1:
        # This can probably be defined, but it isn't yet.
        raise NotImplementedError(interpolations)

    t_units = set(arg.t_unit for arg in args)
    v_units = set(arg.v_unit for arg in args)
    if len(t_units) != 1:
        raise NotImplementedError()
    if len(v_units) != 1:
        raise NotImplementedError()
    t_unit = args[0].t_unit
    v_unit = args[0].v_unit
    vals_by_time = {}
    for arg in args:
        for tt, vv in zip(arg.times, arg.values[1:]):
            vals_by_time.setdefault(tt, []).append(vv)
    times = []
    values = []
    for tt, vvs in sorted(vals_by_time.items()):
        times.append(tt)
        values.append(builtins.sum(vvs))

    rval = SparseTimeSeries(
        times=[tt * t_unit for tt in times],
        values=[vv * v_unit for vv in values],
        default_value=None,
        t_unit=t_unit,
        unit=v_unit,
        identifier=None,
        interpolation='no_interpolation')
    return rval


def scale(self, amount):
    assert isinstance(amount, (float, int))
    if self.interpolation == InterpolationMode.no_interpolation:
        default_value = None
        interpolation = InterpolationMode.no_interpolation
    else:
        raise NotImplementedError()
    rval = SparseTimeSeries(
        times=[tt * self.t_unit for tt in self.times],
        values=[vv * amount * self.v_unit for vv in self.values[1:]],
        default_value=default_value,
        t_unit=self.t_unit,
        unit=self.v_unit,
        identifier=None,
        interpolation=interpolation)
    return rval


def scale_convert(self, amount):
    if self.interpolation == InterpolationMode.no_interpolation:
        default_value = None
        interpolation = InterpolationMode.no_interpolation
    else:
        raise NotImplementedError()
    v_coef = amount * self.v_unit
    rval = SparseTimeSeries(
        times=[tt * self.t_unit for tt in self.times],
        values=[vv * v_coef for vv in self.values[1:]],
        default_value=default_value,
        t_unit=self.t_unit,
        unit=v_coef.u,
        identifier=None,
        interpolation=interpolation)
    return rval
