import pytest
import numpy as np

from .sts import *
from .ureg import u

nan = float('nan')

def test_add():
    a = annual_report(times=[10 * u.years], values=[5 * u.kg])
    b = annual_report(times=[10 * u.years], values=[8 * u.kg])

    c = a + b
    assert c.times == array.array('d', [10])
    assert np.isnan(c.values[0])
    assert c.values[1:] == array.array('d', [13])
    assert c.v_unit == u.kg

def test_to():
    a = annual_report(times=[10 * u.years], values=[1000 * u.kg]).to('metric_ton')
    print(a.v_unit)
    print(u.metric_ton)
    assert a.v_unit == u.metric_ton
    assert a.values[1] == 1

def test_scale_convert():
    a = scale_convert(
        annual_report(times=[10 * u.years], values=[1000 * u.kg]),
        2 * u.meter / u.kg)
    print(a.v_unit)
    print(u.metric_ton)
    assert a.v_unit == u.meter
    assert a.values[1] == 2000

def test_mul_scale_convert():
    a = (annual_report(times=[10 * u.years], values=[1000 * u.kg]) * (2 * u.meter / u.kg))
    print(a.v_unit)
    print(u.metric_ton)
    assert a.v_unit == u.meter
    assert a.values[1] == 2000


def test_setdefault_zero():
    a = annual_report(times=[10 * u.years], values=[1 * u.m])
    b = with_default_zero(a, [5 * u.years, 10 * u.years, 15 * u.years])
    c = with_default_zero(a, [5 * u.years, 15 * u.years])

    assert b.times == array.array('d', [5, 10, 15])
    assert c.values[1:] == array.array('d', [0, 1, 0])

    assert b.times == c.times
    assert b.values[1:] == c.values[1:]


def test_idx_of_first_len_1():
    a = annual_report(times=[10 * u.years], values=[1 * u.m])
    assert a._idx_of_time(10 * u.years) == (1, True)


def test_idx_of_first_len_2():
    a = annual_report(times=[10 * u.years, 20 * u.years], values=[1 * u.m, 2 * u.m])
    assert a._idx_of_time(10 * u.years) == (1, True)
    assert a._idx_of_time(20 * u.years) == (2, True)


def test_pint_rmul():
    a = u.Quantity('1 kg')
    b = annual_report(times=[10 * u.years, 20 * u.years], values=[1 * u.m, 2 * u.m])
    c = a * b
    assert isinstance(c, STS)
    assert c.v_unit == u.kg * u.m


def test_decay():
    a = annual_report(
        times=[10 * u.years, 12 * u.years],
        values=[1 * u.m, 2 * u.m])
    b = annual_report_decay(
        a,
        timescale=4 * u.years,
        horizon=50 * u.years)

    assert 0 * u.m <= b.max() < 2 * u.m
    assert a.sum() == 3 * u.m
    assert b.values[1] == 0.25
    assert 2.99 * u.m <= b.sum() <= 3 * u.m
