import functools
from .sts import STSDict

def add_stsdict_stsdict(self, other):

    val_d = {}
    my_keys = set(self.val_d.keys())
    other_keys = set(other.val_d.keys())
    result_keys = my_keys | other_keys
    for key in result_keys:
        my_val = self.val_d.get(key)
        other_val = other.val_d.get(key)
        if my_val is None and other_val is None:
            val_d[key] = None
        elif my_val is None:
            if self.assume_None_is_zero:
                val_d[key] = other_val
            else:
                raise KeyError(key)
        elif other_val is None:
            if other.assume_None_is_zero:
                val_d[key] = my_val
            else:
                raise KeyError(key)
        else:
            val_d[key] = my_val + other_val
    return self.__class__(
        val_d=val_d,
        assume_None_is_zero=self.assume_None_is_zero and other.assume_None_is_zero)


def mul_stsdict_stsdict(aa, bb):
    assert aa.dims == bb.dims
    dims = aa.dims
    broadcast = [bca and bcb for bca, bcb in zip(aa.broadcast, bb.broadcast)]
    nbdims = [dim for dim, bc in zip(dims, broadcast) if not bc]

    aa_key_pos_list = []
    bb_key_pos_list = []

    # make it dense :/
    val_d = {}
    for key in functools.product(nbdims):
        key_aa = [key[ii] for ii in aa_key_pos_list]
        key_bb = [key[ii] for ii in bb_key_pos_list]
        val_aa = aa.nonbroadcast_getitem(key_aa)
        val_bb = aa.nonbroadcast_getitem(key_bb)
        val_d[key] = val_aa * val_bb

    return STSDict(
        val_d=val_d,
        dims=dims,
        broadcast=broadcast)



class DimensionalityMismatch(Exception):
    pass


class MatMul(object):
    def __init__(self, stsd, dim_idx):
        self.stsd = stsd
        self.dim_idx = dim_idx
        assert 0 <= self.dim_idx < len(self.stsd.dims)

    @property
    def outer_dims(self):
        rval = list(self.stsd.dims)
        rval.pop(self.dim_idx)
        return rval

    @property
    def outer_broadcast(self):
        rval = list(self.stsd.broadcast)
        rval.pop(self.dim_idx)
        return rval

    @property
    def inner_dim(self):
        return self.stsd.dims[self.dim_idx]

    def inner_outer_val_d_key(self, val_d_key):
        rval = []
        val_d_key_pos = 0

        for ii, bc in enumerate(self.stsd.broadcast):
            if ii == self.dim_idx:
                inner = None if bc else val_d_key[val_d_key_pos]
            if not bc and ii != self.dim_idx:
                rval.append(val_d_key[val_d_key_pos])
            if not bc:
                val_d_key_pos += 1

        return inner, tuple(rval)

    def nbkey_from_inner_outer(self, inner_key, outer_key):
        assert not self.stsd.broadcast[self.dim_idx]
        rval = []
        outer_key_pos = 0
        inner_pos = None
        for ii, bc in enumerate(self.stsd.broadcast):
            if ii == self.dim_idx:
                inner_pos = len(rval)
                rval.append(inner_key)
            elif not bc:
                rval.append(outer_key[outer_key_pos])
                outer_key_pos += 1
        return rval, inner_pos

    def elems_by_outer_key(self):
        rval = {}
        for val_d_key, val in self.stsd.val_d.items():
            inner_key, outer_key = self.inner_outer_val_d_key(val_d_key)
            if inner_key is None:
                # broadcasting
                rval[outer_key] = val
            else:
                rval.setdefault(outer_key, {})[inner_key] = val
        # fill out dicts with missing elements from the fallback
        if self.stsd.fallback is not None:
            fallback = self.stsd.fallback
            inner_dim = self.inner_dim
            for outer_key, ikd in rval.items():
                if not isinstance(ikd, dict):
                    continue
                if len(ikd) < len(inner_dim):
                    if fallback.broadcast[self.dim_idx]:
                        # get this value and then fill out ikd with it
                        foo = [
                            fbc
                            for ii, (bc, fbc) in enumerate(zip(self.stsd.broadcast, fallback.broadcast))
                            if not bc and ii != self.dim_idx]
                        assert len(foo) == len(outer_key)
                        fb_outer_key = [
                            key_elem
                            for key_elem, fbc in zip(outer_key, foo)
                            if not fbc]
                        val = fallback.nonbroadcast_getitem(fb_outer_key)
                        for dim_elem in self.inner_dim:
                            ikd.setdefault(dim_elem, val)
                    else:
                        assert not self.stsd.broadcast[self.dim_idx]
                        nb_key, pos = self.nbkey_from_inner_outer(None, outer_key)
                        for dim_elem in self.inner_dim:
                            if dim_elem in ikd:
                                continue
                            nb_key[pos] = dim_elem
                            ikd[dim_elem] = self.stsd.nonbroadcast_getitem(nb_key)
        return rval


def matmul_stsdict_stsdict(self, other, self_dim=None, other_dim=None):
    if self_dim is None:
        self_dim = len(self.dims) - 1
    if other_dim is None:
        other_dim = 0
    mma = MatMul(self, self_dim)
    mmb = MatMul(other, other_dim)

    if mma.inner_dim != mmb.inner_dim:
        raise DimensionalityMismatch(mma.inner_dim, mmb.inner_dim)
    inner_dim = mma.inner_dim

    val_d = {}
    summed_keys = {}
    for a_outer_key, a_vals in mma.elems_by_outer_key().items():
        for b_outer_key, b_vals in mmb.elems_by_outer_key().items():
            sumprod = None
            assert len(inner_dim)
            block = None
            if isinstance(a_vals, dict) and isinstance(b_vals, dict):
                block = 'a'
                for inner_key in inner_dim:
                    a_val = a_vals[inner_key]
                    b_val = b_vals[inner_key]
                    if sumprod is None:
                        sumprod = a_val * b_val
                    else:
                        try:
                            sumprod += a_val * b_val
                        except Exception as exc:
                            raise RuntimeError(a_outer_key, b_outer_key, inner_key, sumprod, a_val, b_val) from exc
            elif isinstance(b_vals, dict):
                block = 'b'
                a_val = a_vals
                for inner_key in inner_dim:
                    b_val = b_vals[inner_key]
                    if sumprod is None:
                        sumprod = a_val * b_val
                    else:
                        sumprod += a_val * b_val
            else:
                raise NotImplementedError(a_vals, b_vals)
            assert sumprod is not None, (block, a_outer_key, a_vals, b_outer_key, b_vals)
            val_d[a_outer_key + b_outer_key] = sumprod
    if self.fallback is not None and other.fallback is not None:
        fallback = self.fallback @ other.fallback
    elif self.fallback is not None and len(other.val_d) == other.logical_size:
        fallback = self.fallback @ other
    else:
        fallback = None
    return STSDict(
        val_d=val_d,
        dims=mma.outer_dims + mmb.outer_dims,
        broadcast=mma.outer_broadcast + mmb.outer_broadcast,
        fallback=fallback)
