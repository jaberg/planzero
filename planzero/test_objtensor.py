import enum
from .objtensor import *
from .ureg import u


class A(str, enum.Enum):
    A = 'A'
    B = 'B'

class B(str, enum.Enum):
    A = 'A'
    B = 'B'
    C = 'C'


def test_views():
    foo = empty(A)
    assert foo.buf == [None, None]

    foo[A.A] = 0
    assert foo.buf == [0, None]

    foo[A.B] = 1
    assert foo.buf == [0, 1]

    bar = empty([A.B, A.A])
    assert foo.dims != bar.dims
    assert dims_are_ravel_compatible(foo.dims, bar.dims)

    bar[A.A] = 0
    assert bar.buf == [None, 0]

    bar[A.B] = 2
    assert bar.buf == [2, 0]

    foo.update(bar)
    assert foo.buf == [0, 2]


def test_getitem_helper_full():
    foo = empty(A)
    keep_dims, view_dims, offset, keep_strides, view_strides = foo.getitem_helper(slice(None))
    assert keep_dims == foo.dims
    assert view_dims == []
    assert offset == foo.offset
    assert keep_strides == foo.strides
    assert view_strides == []

def test_fill_1():
    foo = empty(A)
    foo.fill('a')
    assert all([aa == 'a' for aa in foo.buf])

def test_view_fill_1():
    foo = empty(A)
    bar = foo[:]
    bar.fill('a')
    assert foo.dims == bar.dims
    assert foo.strides == bar.strides
    assert foo.offset == bar.offset
    assert foo.buf is bar.buf
    assert all([aa == 'a' for aa in foo.buf])


def test_fill_by_slice():
    foo = empty(A)
    bar = empty(A, B)
    foo[:] = 1
    assert all([aa == 1 for aa in foo.buf])

    q = 42 * u.MG / u.m3_NG_nmk
    bar[:] = q
    assert all([aa == q for aa in bar.buf])


def test_sum_dim():
    bar = empty(A, B)
    bar.buf[:6] = range(6)
    # 0 1 2
    # 3 4 5

    bar_sum_0 = bar.sum(0)
    assert bar_sum_0.dims == [{B.A: 0, B.B: 1, B.C: 2}]
    assert bar_sum_0.buf == [3, 5, 7]

    bar_sum_1 = bar.sum(1)
    assert bar_sum_1.dims == [{A.A: 0, A.B: 1}]
    assert bar_sum_1.buf == [3, 12]

def test_sum_dim3():
    bar = empty(A, A, B)
    bar.buf[:] = range(12)
    # 0 1 2
    # 3 4 5
    #
    # 6 7 8
    # 9 10 11

    bar_sum_0 = bar.sum(0)
    assert bar_sum_0.dims == empty(A, B).dims
    assert bar_sum_0.buf == [6, 8, 10, 12, 14, 16]

    bar_sum_1 = bar.sum(1)
    assert bar_sum_1.dims == empty(A, B).dims
    assert bar_sum_1.buf == [3, 5, 7, 15, 17, 19]

    bar_sum_2 = bar.sum(2)
    assert bar_sum_2.dims == empty(A, A).dims
    assert bar_sum_2.buf == [3, 12, 21, 30]
