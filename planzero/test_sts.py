import pytest
import numpy as np

from .sts import *
from .stsdict_fns import *
from .ureg import u
from .enums import Geo, GHG

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


def test_usum():
    a = annual_report(times=[10 * u.years], values=[1 * u.m])
    b = annual_report(times=[20 * u.years], values=[2 * u.m])
    c = annual_report(times=[10 * u.years], values=[3 * u.m])

    s = usum([a, b, c, None])

    assert s.times == array.array('d', [10, 20])
    assert s.values[1:] == array.array('d', [4, 2])


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
    assert isinstance(c, SparseTimeSeries)
    assert c.v_unit == u.kg * u.m


def test_matmul_missing():
    a_val_d = {
        (Geo.AB, GHG.CO2): 1.0,
        (Geo.BC, GHG.CO2): 2.0,
        (Geo.AB, GHG.CH4): 3.0,
        (Geo.BC, GHG.CH4): 4.0,
        (Geo.SK, GHG.CH4): 5.0,
    }
    # a large sparse matrix with 6 elements
    a = STSDict(
        val_d=a_val_d,
        dims=[Geo, GHG],
        broadcast=[False, False],
        fallback=None)

    b_val_d = {
        GHG.CO2: 1.0,
        GHG.CH4: 3.0,
    }
    b = STSDict(
        val_d=b_val_d,
        dims=[GHG],
        broadcast=[False],
        fallback=None)

    with pytest.raises(KeyError):
        # missing lots of keys
        ab = a @ b

def test_matmul_dim_mismatch():
    a_val_d = {
        (Geo.AB, GHG.CO2): 1.0,
        (Geo.BC, GHG.CO2): 2.0,
        (Geo.AB, GHG.CH4): 3.0,
        (Geo.BC, GHG.CH4): 4.0,
        (Geo.SK, GHG.CH4): 5.0,
    }
    # sparse with regards to Geo
    # but now dense with regards to GHG
    a = STSDict(
        val_d=a_val_d,
        dims=[Geo, [GHG.CO2, GHG.CH4]],
        broadcast=[False, False],
        fallback=None)

    b_val_d = {
        GHG.CO2: 1.0,
        GHG.CH4: 3.0,
    }
    b = STSDict(
        val_d=b_val_d,
        dims=[GHG],
        broadcast=[False],
        fallback=None)

    with pytest.raises(DimensionalityMismatch):
        # missing lots of keys
        ab = a @ b

def test_matmul_mat_vec():
    a_val_d = {
        (Geo.AB, GHG.CO2): 1.0,
        (Geo.BC, GHG.CO2): 2.0,
        (Geo.AB, GHG.CH4): 3.0,
        (Geo.BC, GHG.CH4): 4.0,
        (Geo.SK, GHG.CH4): 5.0,
    }
    # sparse with regards to Geo
    # but now dense with regards to GHG
    a = STSDict(
        val_d=a_val_d,
        dims=[Geo, [GHG.CO2, GHG.CH4]],
        broadcast=[False, False],
        fallback=STSDict(
            val_d={(): 9.0},
            dims=[Geo, [GHG.CO2, GHG.CH4]],
            broadcast=[True, True],
            fallback=None
            ))

    b_val_d = {
        GHG.CO2: 1.0,
        GHG.CH4: 3.0,
    }
    b = STSDict(
        val_d=b_val_d,
        dims=[[GHG.CO2, GHG.CH4]],
        broadcast=[False],
        fallback=None)

    ab = a @ b
    assert len(ab.val_d) == 3
    assert ab[Geo.AB] == 10
    assert ab[Geo.BC] == 14
    assert ab[Geo.SK] == 24

def test_matmul_mat_vec_fallback_semibroadcasting():
    a_val_d = {
        (Geo.AB, GHG.CO2): 1.0,
        (Geo.BC, GHG.CO2): 2.0,
        (Geo.AB, GHG.CH4): 3.0,
        (Geo.BC, GHG.CH4): 4.0,
        (Geo.SK, GHG.CH4): 5.0,
        (Geo.MB, GHG.CO2): 6.0,
    }
    # sparse with regards to Geo
    # but now dense with regards to GHG
    a = STSDict(
        val_d=a_val_d,
        dims=[Geo, [GHG.CO2, GHG.CH4]],
        broadcast=[False, False],
        fallback=STSDict(
            val_d={(GHG.CO2,): 7.0, (GHG.CH4,): 8.0},
            dims=[Geo, [GHG.CO2, GHG.CH4]],
            broadcast=[True, False],
            fallback=None
            ))
    mma = MatMul(a, 1)
    mma_elems_by_outer_key = mma.elems_by_outer_key()
    assert mma_elems_by_outer_key == {
        (Geo.AB,): {GHG.CO2: 1.0, GHG.CH4: 3.0},
        (Geo.BC,): {GHG.CO2: 2.0, GHG.CH4: 4.0},
        (Geo.SK,): {GHG.CO2: 7.0, GHG.CH4: 5.0},
        (Geo.MB,): {GHG.CO2: 6.0, GHG.CH4: 8.0},
    }

    b_val_d = {
        GHG.CO2: 1.0,
        GHG.CH4: 3.0,
    }
    b = STSDict(
        val_d=b_val_d,
        dims=[[GHG.CO2, GHG.CH4]],
        broadcast=[False],
        fallback=None)

    mmb = MatMul(b, 0)
    mmb_elems_by_outer_key = mmb.elems_by_outer_key()
    assert mmb_elems_by_outer_key == {
        (): {GHG.CO2: 1.0, GHG.CH4: 3.0},
    }

    ab = a @ b
    assert len(ab.val_d) == 4
    assert ab[Geo.AB] == 1 * 1 + 3 * 3
    assert ab[Geo.BC] == 2 * 1 + 4 * 3
    assert ab[Geo.SK] == 7 * 1 + 5 * 3
    assert ab[Geo.MB] == 6 * 1 + 8 * 3
    assert ab[Geo.ON] == 7 * 1 + 8 * 3
    assert ab[Geo.QC] == 7 * 1 + 8 * 3
