import array
import bisect
from enum import Enum

import numpy as np
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
                 interpolation='current'):
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
            self.extend(times, values)

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
                    valid = self.times[index] == ts
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

    def extend(self, times, values):
        assert len(times) == len(values)
        for t, v in zip(times, values):
            self.append(t, v)

    def plot(self, t_unit=None, annotate=True, **kwargs):
        # XXX This is buggy re: T_unit
        # also re: interpolation
        t_unit = t_unit or self.t_unit
        import matplotlib.pyplot as plt
        plt.scatter(
            self.times,
            self.values[1:],
            **kwargs)
        plt.xlabel(t_unit)
        plt.ylabel(self.v_unit)
        plt.title(self.identifier)
        if annotate:
            self.annotate_plot(t_unit=t_unit, **kwargs)

    def annotate_plot(self, t_unit=None, **kwargs):
        """Called once per variable name in comparison plots"""
        pass

    def __truediv__(self, other):
        default_value = (
            None
            if self.interpolation is InterpolationMode.no_interpolation
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
        raise NotImplementedError()

def annual_summary(**kwargs):
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
