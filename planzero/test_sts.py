import numpy as np

from .sts import *
from .ureg import u

nan = float('nan')

def test_usum():
    a = annual_report(times=[10 * u.years], values=[5 * u.kg])
    b = annual_report(times=[10 * u.years], values=[8 * u.kg])

    c = usum([a, b])
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
