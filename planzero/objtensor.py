import enum
import itertools

import numpy as np
import pint
from pydantic import BaseModel

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


def wrap_in_n_lists(x, n):
    if n == 0:
        return x
    else:
        return [wrap_in_n_lists(x, n - 1)]


class DimsMismatch(Exception):
    pass


class ObjectTensor(BaseModel):
    dims: list[dict[object, int] | None]
    npr: object

    def __init__(self, *, dims, npr):
        super().__init__(dims=dims, npr=npr)
        assert self.shape == self.npr.shape
        try:
            self.npr.u
            # pint is sneaking in, we want per-element objects
            self.npr = np.asarray(list(self.npr.ravel()), dtype='object').reshape(self.npr.shape)
            assert not hasattr(self.npr, 'u')
        except AttributeError:
            pass

    def __iter__(self):
        for npr_i in self.npr:
            if len(self.dims) > 1:
                yield ObjectTensor(dims=self.dims[1:], npr=npr_i)
            else:
                yield npr_i

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
        return self.npr.ravel()

    def ravel_items(self):
        if None in self.dims:
            raise NotImplementedError()
        no_none_dims = [dim for dim in self.dims]
        for key, val in zip(itertools.product(*no_none_dims), self.npr.ravel()):
            yield key, val

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
        shape = [len(dim) for dim in dims]
        npr = np.empty(shape, dtype='object')
        assert not hasattr(npr, 'dims')
        return cls(dims=dims, npr=npr)

    @classmethod
    def from_dict(cls, dct):
        dims = cls.as_dims([dct])
        npr = np.asarray(list(dct.values()), dtype='object')
        return cls(dims=dims, npr=npr)

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
        idx = []
        view_dims = list(self.dims)
        keep_dims = []
        # This iteration may run short, that's okay
        for item_i in item:
            if item_i == slice(None):
                idx.append(item_i)
                dim_i = view_dims.pop(0)
                keep_dims.append(dim_i)
            elif item_i is None:
                idx.append(None)
                keep_dims.append(None)
            elif _issubclass(item_i, enum.Enum):
                dim_i = view_dims.pop(0)
                idx.append([dim_i[item_ij] for item_ij in item_i])
                keep_dims.append({item_ij: ii for ii, item_ij in enumerate(item_i)})
            elif isinstance(item_i, (list, tuple, dict)):
                dim_i = view_dims.pop(0)
                idx.append([dim_i[item_ij] for item_ij in item_i])
                keep_dims.append({item_ij: ii for ii, item_ij in enumerate(item_i)})
            else:
                try:
                    dim_i = view_dims.pop(0)
                    idx.append(dim_i[item_i])
                except IndexError:
                    raise NotImplementedError()
        n_wrapping_lists = 0
        for ii in range(len(idx) - 1, -1, -1):
            if isinstance(idx[ii], (int, slice)):
                continue
            elif idx[ii] is None:
                continue
            else:
                # make them broadcast
                idx[ii] = [wrap_in_n_lists(idx_ij, n_wrapping_lists)
                           for idx_ij in idx[ii]]
                n_wrapping_lists += 1
        return idx, keep_dims, view_dims

    def __getitem__(self, item):
        idx, keep_dims, view_dims = self.getitem_helper(item)
        if view_dims or keep_dims:
            npr = self.npr[*idx]
            return ObjectTensor(dims=keep_dims + view_dims, npr=npr)
        else:
            return self.npr[*idx]
    
    def __setitem__(self, item, value):
        idx, keep_dims, view_dims = self.getitem_helper(item)
        if keep_dims or view_dims:
            if isinstance(value, pint.Quantity):
                # broadcasting scalars seemed broken, units were lost
                dims = keep_dims + view_dims
                shape = shape_from_dims(dims)
                assert self.npr[*idx].shape == shape, (self.npr.shape, idx, self.npr[*idx].shape, shape)
                broadcasted_value = np.asarray(
                    [value] * size_from_dims(dims),
                    dtype='object').reshape(shape)
                self.npr[*idx] = broadcasted_value
            elif isinstance(value, ObjectTensor):
                dims = view_dims + keep_dims
                if dims == value.dims:
                    self.npr[*idx] = value.npr
                else:
                    raise NotImplementedError()
            else:
                # trust broadcasting assignment
                self.npr[*idx] = value
        else:
            assert not isinstance(value, ObjectTensor)
            self.npr[*idx] = value

    def __mul__(self, other):
        if isinstance(other, ObjectTensor):
            # check compatibility
            if len(self.dims) > len(other.dims):
                a_dims = self.dims
                b_dims = [None] * (len(self.dims) - len(other.dims)) + other.dims
            elif len(self.dims) < len(other.dims):
                a_dims = [None] * (len(other.dims) - len(self.dims)) + self.dims
                b_dims = other.dims
            else:
                a_dims = self.dims
                b_dims = other.dims

            r_dims = []
            for a_dim, b_dim in zip(a_dims, b_dims):
                if a_dim is None and b_dim is None:
                    r_dims.append(None)
                elif a_dim is None and b_dim is not None:
                    r_dims.append(b_dim)
                elif a_dim is not None and b_dim is None:
                    r_dims.append(a_dim)
                else:
                    if a_dim == b_dim:
                        r_dims.append(a_dim)
                    elif set(a_dim) == set(b_dim):
                        raise NotImplementedError(a_dim, b_dim)
                    else:
                        raise DimsMismatch(a_dim, b_dim)
            npr = self.npr * other.npr
            return ObjectTensor(dims=r_dims, npr=npr)
        else:
            return ObjectTensor(dims=self.dims, npr=self.npr * other)

    def __truediv__(self, other):
        if isinstance(other, ObjectTensor):
            # check compatibility
            if len(self.dims) > len(other.dims):
                a_dims = self.dims
                b_dims = [None] * (len(self.dims) - len(other.dims)) + other.dims
            elif len(self.dims) < len(other.dims):
                a_dims = [None] * (len(other.dims) - len(self.dims)) + self.dims
                b_dims = other.dims
            else:
                a_dims = self.dims
                b_dims = other.dims

            r_dims = []
            for a_dim, b_dim in zip(a_dims, b_dims):
                if a_dim is None and b_dim is None:
                    r_dims.append(None)
                elif a_dim is None and b_dim is not None:
                    r_dims.append(b_dim)
                elif a_dim is not None and b_dim is None:
                    r_dims.append(a_dim)
                else:
                    if a_dim == b_dim:
                        r_dims.append(a_dim)
                    elif set(a_dim) == set(b_dim):
                        raise NotImplementedError(a_dim, b_dim)
                    else:
                        raise DimsMismatch(a_dim, b_dim)
            npr = self.npr / other.npr
            return ObjectTensor(dims=r_dims, npr=npr)
        else:
            return ObjectTensor(dims=self.dims, npr=self.npr / other)

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

    def sum(self, sum_dim=None):
        if sum_dim is None:
            return self.npr.sum()
        elif isinstance(sum_dim, int):
            sum_dim_idx = sum_dim
            dims = list(self.dims)
            dims.pop(sum_dim_idx)
            npr = self.npr.sum(sum_dim_idx)
            if isinstance(npr, np.ndarray):
                return ObjectTensor(dims=dims, npr=npr)
            else:
                return npr
        else:
            sum_dim_d = {item: ii for ii, item in enumerate(sum_dim)}
            for ii, dim in enumerate(self.dims):
                if dim == sum_dim_d:
                    return self.sum(ii)
            raise IndexError(sum_dim)

    def t_units(self):
        t_units = set()
        for obj in self.npr.ravel():
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
        for obj in self.npr.ravel():
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
