import numpy as np
import pint
from pydantic import BaseModel
from .ureg import Geo
import matplotlib.pyplot as plt
from . import sts

PTDim = tuple(Geo.provinces_and_territories())


def PTValues(val_d, broadcast=(False,), **kwargs):
    if Geo.CA in val_d:
        ca = val_d.get(Geo.CA)
        px = val_d.get(Geo.PX)
        assert px is None, 'cannot update PX from CA, PX already present'

        pt_val_d = dict(val_d)
        del pt_val_d[Geo.CA]
        pt_usum = sts.usum(pt_val_d.values())
        ca_val = val_d[Geo.CA]
        px_val = ca_val - pt_usum
        pt_val_d[Geo.PX] = px_val
    else:
        pt_val_d = val_d
    return sts.STSDict(
        val_d=pt_val_d,
        dims=[PTDim],
        broadcast=broadcast,
        **kwargs) # may include e.g. fallback


def update_CA(self, overwrite=False):
    if Geo.CA in self.val_d and not overwrite:
        self.val_d[Geo.CA] = sts.usum(self.val_d.values())
    elif Geo.CA in self.val_d:
        assert 0, 'pass overwrite=True'


def national_total(self):
    return sts.usum(self.val_d.values())


def scatter(self, **kwargs):
    t_unit = self.t_unit
    v_unit = self.v_unit
    for key, val in self.val_d.items():
        if val is None:
            pass
        elif isinstance(val, pint.Quantity):
            plt.axhline(val.magnitude)
        elif val.times:
            plt.scatter(
                val.times,
                val.values[1:],
                label=key,
                **kwargs)
    plt.xlabel(t_unit)
    plt.ylabel(v_unit)

def stacked_bar(self, **kwargs):
    t_unit = self.t_unit
    v_unit = self.v_unit
    bottom = None
    for key, val in self.val_d.items():
        if val is None:
            continue
        elif isinstance(val, pint.Quantity):
            raise NotImplementedError()
        else:
            plt.bar(
                val.times,
                val.values[1:],
                label=key,
                bottom=0 if bottom is None else bottom,
                **kwargs)
        if bottom is None:
            bottom = np.asarray(val.values[1:])
        else:
            bottom += np.asarray(val.values[1:])
    plt.xlabel(t_unit)
    plt.ylabel(v_unit)
