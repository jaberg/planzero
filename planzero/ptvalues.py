import numpy as np
import pint
from pydantic import BaseModel
from .ureg import Geo
import matplotlib.pyplot as plt
from . import sts

_provinces_and_territories = frozenset(Geo.provinces_and_territories())


class PTValues(BaseModel):
    val_d: dict[Geo, object | None]

    def __getitem__(self, key):
        return self.val_d[key]

    def __setitem__(self, key, value):
        assert key in _provinces_and_territories
        self.val_d[key] = value

    def __add__(self, other):
        if isinstance(other, PTValues):
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
                    val_d[key] = other_val
                elif other_val is None:
                    val_d[key] = my_val
                else:
                    val_d[key] = my_val + other_val
            return PTValues(val_d=val_d)
        else:
            # too much ambiguity about when there are key=None entries
            raise NotImplementedError()

    def __mul__(self, other):
        if isinstance(other, PTValues):
            val_d = {}
            my_keys = set(self.val_d.keys())
            other_keys = set(other.val_d.keys())
            result_keys = my_keys & other_keys
            for key in result_keys:
                my_val = self.val_d.get(key)
                other_val = other.val_d.get(key)
                if my_val is None and other_val is None:
                    val_d[key] = None
                elif my_val is None:
                    val_d[key] = None
                elif other_val is None:
                    val_d[key] = None
                else:
                    val_d[key] = my_val * other_val
            return PTValues(val_d=val_d)
        else:
            val_d = {}
            for key, val in self.val_d.items():
                if val is None:
                    val_d[key] = None
                else:
                    val_d[key] = val * other
            return PTValues(val_d=val_d)

    def to(self, unit):
        return PTValues(val_d={
            pt: (None if val is None else val.to(unit))
            for pt, val in self.val_d.items()})

    @property
    def t_unit(self):
        t_units = set()
        for val in self.val_d.values():
            if val is None:
                continue
            elif isinstance(val, pint.Quantity):
                continue
            else:
                t_units.add(val.t_unit)
        t_unit, = t_units
        return t_unit

    @property
    def v_unit(self):
        v_units = set()
        for val in self.val_d.values():
            if val is None:
                continue
            elif isinstance(val, pint.Quantity):
                v_units.add(val.u)
            else:
                v_units.add(val.v_unit)
        v_unit, = v_units
        return v_unit

    def update_CA(self, overwrite=False):
        if Geo.CA in self.val_d and not overwrite:
            self.val_d[Geo.CA] = sts.usum(self.val_d.values())
        elif Geo.CA in self.val_d:
            assert 0, 'pass overwrite=True'

    def replace_CA_with_PX(self, overwrite=False):
        ca = self.val_d.get(Geo.CA)
        assert ca is not None, 'cannot update PX from CA without CA'
        px = self.val_d.get(Geo.PX)
        assert px is None, 'cannot update PX from CA, PX already present'

        pt_val_d = dict(self.val_d)
        del pt_val_d[Geo.CA]
        pt_usum = sts.usum(pt_val_d.values())

        ca_val = self.val_d[Geo.CA]
        px_val = ca_val - pt_usum
        self.val_d[Geo.PX] = px_val
        del self.val_d[Geo.CA]

    def national_total(self):
        if Geo.CA in self.val_d:
            return self.val_d[Geo.CA]
        else:
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
        if Geo.CA in self.val_d:
            raise NotImplementedError()
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
