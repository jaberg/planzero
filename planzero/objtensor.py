import enum
import itertools

import numpy as np
import pint

def print_dims(msg, dims):
    for ii, dim in enumerate(dims):
        print(msg, ii, dim)

def _issubclass(foo, bar):
    try:
        return issubclass(foo, bar)
    except TypeError:
        return False

def shape_from_dims(dims):
    return tuple([1 if dim is None else len(dim) for dim in dims])


def size_from_dims(dims):
    rval = 1
    for shape_i in shape_from_dims(dims):
        rval *= shape_i
    return rval


def strides_from_dims(dims):
    stride = 1
    strides = []
    for shape_i in reversed(shape_from_dims(dims)):
        strides.append(stride)
        stride *= shape_i
    strides.reverse()
    return strides


def wrap_in_n_lists(x, n):
    if n == 0:
        return x
    else:
        return [wrap_in_n_lists(x, n - 1)]


def key_idx_dot(key_idxs, strides):
    offset = 0
    key = []
    for (key_i, idx_i), stride_i in zip(key_idxs, strides):
        offset += idx_i * stride_i
        key.append(key_i)
    return key, offset


def squeeze_dims_strides(dims, strides, axis=None):
    if axis is None:
        no_none_dims = []
        no_none_strides = []
        for dim_i, stride_i in zip(dims, strides):
            if dim_i is None:
                continue
            else:
                no_none_dims.append(dim_i)
                no_none_strides.append(stride_i)
        return no_none_dims, no_none_strides
    else:
        assert self.dims[axis] is None
        new_dims = list(self.dims)
        new_strides = list(self.strides)
        new_dims.pop(axis)
        new_strides.pop(axis)
        return new_dims, new_strides


def ravel_keys_offsets(dims, strides, offset):
    no_none_dims, no_none_strides = squeeze_dims_strides(dims, strides)
    for key_idx_tuple in itertools.product(*[sorted(dim_i.items()) for dim_i in no_none_dims]):
        key, idx_offset = key_idx_dot(key_idx_tuple, no_none_strides)
        yield tuple(key), (offset + idx_offset)


def ravel_multi(*ots):
    for key_offs in zip(*[ot.ravel_keys_offsets() for ot in ots]):
        keys, offs = zip(*key_offs)
        key, = set(keys)
        yield key, offs


def dims_are_ravel_compatible(aa_dims, bb_dims, cc_dims=None):
    if cc_dims is None:
        cc_dims = bb_dims
    for aa_dim_i, bb_dim_i, cc_dim_i in zip(aa_dims, bb_dims, cc_dims):
        if aa_dim_i is None and bb_dim_i is None and cc_dim_i is None:
            continue
        if set(aa_dim_i) == set(bb_dim_i) == set(cc_dim_i):
            continue
        return False
    return True


def elemwise_binary_op_dims(a_dims, b_dims):
    if len(b_dims) > len(a_dims):
        return elemwise_binary_op_dims(b_dims, a_dims)
    # now assume length of a_dims >= length of b_dims
    b_dims = [None] * (len(a_dims) - len(b_dims)) + b_dims
    assert len(a_dims) == len(b_dims)

    r_dims = []
    for a_dim, b_dim in zip(a_dims, b_dims):
        if a_dim is None and b_dim is None:
            r_dims.append(None)
        elif a_dim is None and b_dim is not None:
            r_dims.append(b_dim)
        elif a_dim is not None and b_dim is None:
            r_dims.append(a_dim)
        else:
            if set(a_dim) == set(b_dim):
                r_dims.append(a_dim)
            else:
                raise DimsMismatch(a_dim, b_dim)
    return r_dims


class DimsMismatch(Exception):
    pass


class ObjectTensor(object):

    # Pydantic BaseModel was slightly comforting, but
    # I believe it caused buf to be copied on __init__ which breaks views
    # TODO: work around that somehow?
    dims: list[dict[object, int] | None]
    strides: list[int]
    offset: int
    buf: list[object]

    def __init__(self, *, dims, strides, offset, buf):
        self.dims = dims
        self.strides = strides
        self.offset = offset
        self.buf = buf
        #super().__init__(dims=dims, strides=strides, offset=offset, buf=buf)
        assert len(strides) == len(dims)

    def copy(self):
        rval = ObjectTensor.empty(*self.dims)
        rval.update(self)
        return rval

    @property
    def obj(self):
        assert dims == []
        return buf[offset]

    def __iter__(self):
        for dim_i in self.dims[0]:
            yield self[dim_i]

    def __len__(self):
        return len(self.dims[0])

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def shape(self):
        return shape_from_dims(self.dims)

    @property
    def size(self):
        return size_from_dims(self.dims)

    def ravel(self):
        for key, off in self.ravel_keys_offsets():
            yield self.buf[off]

    @staticmethod
    def as_dims(dims):
        unpacked_dims = []
        for dim in dims:
            if dim is None:
                unpacked_dims.append(None)
            else:
                unpacked_dims.append({dimval:ii for ii, dimval in enumerate(dim)})
        return unpacked_dims

    @classmethod
    def empty(cls, *dims):
        dims = cls.as_dims(dims)
        shape = shape_from_dims(dims)
        size = size_from_dims(dims)
        strides = strides_from_dims(dims)
        buf = [None] * size
        return cls(dims=dims, strides=strides, offset=0, buf=buf)

    @classmethod
    def from_dict(cls, dct):
        dims = cls.as_dims([dct])
        shape = shape_from_dims(dims)
        size = size_from_dims(dims)
        strides = strides_from_dims(dims)
        buf = list(dct.values())
        return cls(dims=dims, strides=strides, offset=0, buf=buf)

    def getitem_helper(self, item):
        if isinstance(item, int):
            item = item,
        elif isinstance(item, slice):
            item = item,
        elif _issubclass(item, enum.Enum):
            item = item,
        elif isinstance(item, enum.Enum):
            item = item,
        elif isinstance(item, list):
            item = item,
        elif item is None:
            item = item,
        #idx = []
        keep_dims = []
        view_dims = list(self.dims)
        offset = self.offset
        keep_strides = []
        view_strides = list(self.strides)

        for item_i in item:
            if item_i == slice(None):
                #idx.append(item_i)
                dim_i = view_dims.pop(0)
                stride_i = view_strides.pop(0)
                keep_dims.append(dim_i)
                keep_strides.append(stride_i)
            elif item_i is None:
                #idx.append(None)
                keep_dims.append(None)
                keep_strides.append(0)
            elif _issubclass(item_i, enum.Enum):
                dim_i = view_dims.pop(0)
                stride_i = view_strides.pop(0)
                #idx.append([dim_i[item_ij] for item_ij in item_i]) # stride multipliers
                keep_dims.append({item_ij: dim_i[item_ij] for item_ij in item_i}) # dim_i, restricted to the enum elements in item_i
                keep_strides.append(stride_i)
            elif isinstance(item_i, (list, tuple, dict)):
                dim_i = view_dims.pop(0)
                stride_i = view_strides.pop(0)
                #idx.append([dim_i[item_ij] for item_ij in item_i])
                keep_dims.append({item_ij: dim_i[item_ij] for item_ij in item_i})
                keep_strides.append(stride_i)
            else:
                dim_i = view_dims.pop(0)
                stride_i = view_strides.pop(0)
                #idx.append(dim_i[item_i])
                offset += stride_i * dim_i[item_i]
        return keep_dims, view_dims, offset, keep_strides, view_strides

    def _buf_idx(self, idx):
        assert len(idx) == len(self.strides)
        rval = self.offset
        for idx_i, stride_i in zip(idx, self.strides):
            rval += idx_i * stride_i
        return rval

    def __getitem__(self, item):
        keep_dims, view_dims, offset, keep_strides, view_strides = self.getitem_helper(item)
        if keep_dims or view_dims:
            return ObjectTensor(
                dims=keep_dims + view_dims,
                offset=offset,
                strides=keep_strides + view_strides,
                buf=self.buf)
        else:
            return self.buf[offset]

    def broadcast_to_dims(self, target_dims):
        # similar to unsqueeze, but pulling in leading axes from dims.
        # also creates self-aliased view
        n_dims_required = len(target_dims) - len(self.dims)
        if n_dims_required < 0:
            raise DimsMismatch(self.dims, target_dims)
        if n_dims_required:
            dims = target_dims[:n_dims_required] + self.dims
            strides = [0] * n_dims_required + self.strides
        else:
            dims = list(self.dims)
            strides = list(self.strides)

        # now expand None dims to make things work
        for ii in range(len(dims)):
            if dims[ii] is None and target_dims[ii] is not None:
                dims[ii] = target_dims[ii]
                strides[ii] = 0 # self-aliasing trick

        if dims_are_ravel_compatible(target_dims, dims):
            return ObjectTensor(dims=dims, strides=strides, offset=self.offset, buf=self.buf)
        else:
            raise DimsMismatch(self.dims, target_dims)

    def ravel_keys_offsets(self):
        return ravel_keys_offsets(self.dims, self.strides, self.offset)

    def update(self, other):
        other = other.broadcast_to_dims(self.dims)
        for key, (my_offset, other_offset) in ravel_multi(self, other):
            self.buf[my_offset] = other.buf[other_offset]

    def fill(self, value):
        for key, off in self.ravel_keys_offsets():
            self.buf[off] = value
    
    def __setitem__(self, item, value):
        keep_dims, view_dims, offset, keep_strides, view_strides = self.getitem_helper(item)

        dims = keep_dims + view_dims
        strides = keep_strides + view_strides

        if dims:
            lhs = ObjectTensor(dims=dims, strides=strides, offset=offset, buf=self.buf)
            if isinstance(value, ObjectTensor):
                # multi-dimensional dimension-matching
                lhs.update(value)
            elif isinstance(value, dict):
                # single-dimensional dimension-matching
                # keys must match last dimension
                rhs = ObjectTensor.from_dict(value)
                lhs.update(rhs)
            elif isinstance(value, np.ndarray):
                raise NotImplementedError()
            elif isinstance(value, (list, tuple)):
                raise NotImplementedError()
            else:
                # no dimension-matching, treat value as single object
                lhs.fill(value)
        else:
            assert not isinstance(value, ObjectTensor), 'you are probably confused'
            self.buf[offset] = value

    def __mul__(self, other):
        if isinstance(other, ObjectTensor):
            r_dims = elemwise_binary_op_dims(self.dims, other.dims)
            rval = ObjectTensor.empty(*r_dims)
            up_self = self.broadcast_to_dims(r_dims)
            up_other = other.broadcast_to_dims(r_dims)
            for key, (aa, bb, cc) in ravel_multi(rval, up_self, up_other):
                rval.buf[aa] = up_self.buf[bb] * up_other.buf[cc]
            return rval
        else:
            rval = ObjectTensor.empty(*self.dims)
            for key, (aa, bb) in ravel_multi(rval, self):
                rval.buf[aa] = self.buf[bb] * other
            return rval

    def __truediv__(self, other):
        if isinstance(other, ObjectTensor):
            r_dims = elemwise_binary_op_dims(self.dims, other.dims)
            rval = ObjectTensor.empty(*r_dims)
            up_self = self.broadcast_to_dims(r_dims)
            up_other = other.broadcast_to_dims(r_dims)
            for key, (aa, bb, cc) in ravel_multi(rval, up_self, up_other):
                rval.buf[aa] = up_self.buf[bb] / up_other.buf[cc]
            return rval
        else:
            rval = ObjectTensor.empty(*self.dims)
            for key, (aa, bb) in ravel_multi(rval, self):
                rval.buf[aa] = self.buf[bb] / other
            return rval

    def __matmul__(self, other):
        if self.ndim == 0 or other.ndim == 0:
            return self * other
        elif self.ndim == 1 and other.ndim == 1:
            return (self * other).sum()
        elif self.ndim == 1 and other.ndim == 2:
            return (self[:, None] * other).sum(0)
        elif self.ndim == 2 and other.ndim == 1:
            return (self * other).sum(1)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, ObjectTensor):
            r_dims = elemwise_binary_op_dims(self.dims, other.dims)
            rval = ObjectTensor.empty(*r_dims)
            up_self = self.broadcast_to_dims(r_dims)
            up_other = other.broadcast_to_dims(r_dims)
            for key, (aa, bb, cc) in ravel_multi(rval, up_self, up_other):
                rval.buf[aa] = up_self.buf[bb] + up_other.buf[cc]
            return rval
        else:
            rval = ObjectTensor.empty(*self.dims)
            for key, (aa, bb) in ravel_multi(rval, self):
                rval.buf[aa] = self.buf[bb] + other
            return rval

    def squeeze(self, axis=None):
        dims, strides = squeeze_dims_strides(self.dims, self.strides, axis=axis)
        return ObjectTensor(dims=dims, strides=strides, offset=self.offset, buf=self.buf)

    def unsqueeze(self, axis, dim=None):
        # weird operation that you need to create a self-aliased dimension
        # which is used in sum() and other reductions
        dims = list(self.dims)
        strides = list(self.strides)
        dims.insert(axis, dim)
        strides.insert(axis, 0)
        return ObjectTensor(dims=dims, strides=strides, offset=self.offset, buf=self.buf)

    def has_aliased_dim(self):
        for dim_i, stride_i in zip(self.dims, self.strides):
            if dim_i is not None and len(dim_i) > 1 and stride_i == 0:
                return True
        return False

    def sum(self, sum_dim=None):
        if sum_dim is None:
            rval = None
            for key, off in self.ravel_keys_offsets():
                if rval is None:
                    rval = self.buf[off]
                else:
                    rval += self.buf[off]
            return rval

        elif isinstance(sum_dim, int):
            # rename to suggest it's an int, not a dim
            sum_dim_idx = sum_dim
            del sum_dim

            dims = list(self.dims)
            summed_dim = dims.pop(sum_dim_idx)
            if dims:
                rval = ObjectTensor.empty(*dims)
                rval_unsq = rval.unsqueeze(sum_dim_idx, dim=summed_dim)
                assert self.dims == rval_unsq.dims
                for key, (my_offset, rval_offset) in ravel_multi(self, rval_unsq):
                    if rval_unsq.buf[rval_offset] is None:
                        rval_unsq.buf[rval_offset] = self.buf[my_offset]
                    else:
                        rval_unsq.buf[rval_offset] += self.buf[my_offset]
                return rval
            else:
                return self.sum()
        else:
            # find a dimension matching sum_dim and sum that one out
            sum_dim_d = {item: ii for ii, item in enumerate(sum_dim)}
            for ii, dim in enumerate(self.dims):
                if dim == sum_dim_d:
                    return self.sum(ii)
            raise IndexError(sum_dim)

    def t_units(self):
        t_units = set()
        for obj in self.ravel():
            try:
                if obj.t_unit not in t_units:
                    yield obj.t_unit
                    t_units.add(obj.t_unit)
            except (AttributeError, TypeError):
                pass

    def as_one_t_unit(self):
        rval, = set(self.t_units())
        return rval

    def v_units(self):
        v_units = set()
        for obj in self.ravel():
            try:
                if obj.u not in v_units:
                    yield obj.u
                    v_units.add(obj.u)
            except (AttributeError, TypeError):
                try:
                    if obj.v_unit not in v_units:
                        yield obj.v_unit
                        v_units.add(obj.v_unit)
                except (AttributeError, TypeError):
                    pass

    def as_one_v_unit(self):
        rval, = set(self.v_units())
        return rval


empty = ObjectTensor.empty
from_dict = ObjectTensor.from_dict

# sts.py adds SparseTimeSeries to this
_types_for_pint_to_ignore = (ObjectTensor,)

# This seems to work for mul
def _no_pint_operators(value, *args, **kwargs):
    if isinstance(value, _types_for_pint_to_ignore):
        raise TypeError("Skip STS classes")
    return _orig_to_magnitude(value, *args, **kwargs)

# adding this as the above wasn't doing the trick for
# `Quantity + SparseTimeSeries`
def _Quantity_add_sub(self, other, op):
    if isinstance(other, _types_for_pint_to_ignore):
        return NotImplemented
    return _orig_add_sub(self, other, op)


if pint.compat._to_magnitude.__name__ == '_to_magnitude':
    _orig_to_magnitude = pint.compat._to_magnitude
    pint.compat._to_magnitude = _no_pint_operators
    # also replace symbol imports in other modules as required
    pint.facets.plain.quantity._to_magnitude = _no_pint_operators

    _orig_add_sub = pint.Quantity._add_sub
    pint.Quantity._add_sub = _Quantity_add_sub
