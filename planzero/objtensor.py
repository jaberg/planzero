import enum
import numpy as np
import pint
from pydantic import BaseModel


def _issubclass(foo, bar):
    try:
        return issubclass(foo, bar)
    except TypeError:
        return False


class ObjectTensor(BaseModel):
    dims: list[dict[object, int]]
    npr: object

    def __init__(self, *, dims, npr):
        super().__init__(dims=dims, npr=npr)
        assert self.shape == self.npr.shape

    def __iter__(self):
        for npr_i in self.npr:
            if len(self.dims) > 1:
                yield ObjectTensor(dims=self.dims[1:], npr=npr_i)
            else:
                yield npr_i

    @property
    def shape(self):
        return tuple([1 if dim is None else len(dim) for dim in self.dims])

    def ravel(self):
        return self.npr.ravel()

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

    def getitem_helper(self, item):
        if not isinstance(item, (tuple, list)):
            item = item,
        idx = []
        view_dims = list(self.dims)
        keep_dims = []
        # This iteration may run short, that's okay
        for item_i, dim_i in zip(item, self.dims):
            if item_i == slice(None):
                idx.append(item_i)
                view_dims.pop(0)
                keep_dims.append(dim_i)
            elif item_i is None:
                raise NotImplementedError(item_i)
            elif _issubclass(item_i, enum.Enum):
                idx.append(tuple([dim_i[item_ij] for item_ij in item_i]))
                view_dims.pop(0)
                keep_dims.append({item_ij: ii for ii, item_ij in enumerate(item_i)})
            elif isinstance(item_i, (list, tuple, dict)):
                idx.append(tuple([dim_i[item_ij] for item_ij in item_i]))
                view_dims.pop(0)
                keep_dims.append({item_ij: ii for ii, item_ij in enumerate(item_i)})
            else:
                try:
                    idx.append(dim_i[item_i])
                    view_dims.pop(0)
                except IndexError:
                    raise NotImplementedError()
        return idx, view_dims, keep_dims

    def __getitem__(self, item):
        idx, view_dims, keep_dims = self.getitem_helper(item)
        if not keep_dims:
            return self.npr[*idx]
        else:
            npr = self.npr[*idx]
            return ObjectTensor(dims=keep_dims + view_dims, npr=npr)
    
    def __setitem__(self, item, value):
        if not isinstance(item, (tuple, list)):
            item = item,
        idx = []
        # This iteration may run short, that's okay
        for item_i, dim_i in zip(item, self.dims):
            if item_i == slice(None):
                idx.append(item_i)
            else:
                idx.append(dim_i[item_i])
        target_view = self.npr[*idx]

        if isinstance(value, pint.Quantity) and target_view.size > 1:
            # broadcasting scalars seemed broken, units were lost
            broadcasted_value = np.asarray(
                [value] * target_view.size, dtype='object').reshape(target_view.shape)
            self.npr[*idx] = broadcasted_value
        else:
            self.npr[*idx] = value


empty = ObjectTensor.empty
