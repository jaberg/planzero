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
            for key, val in self.val_d.items():
                other_val = other.val_d[key]
                if val is None and other_val is None:
                    val_d[key] = None
                elif val is None:
                    val_d[key] = other_val
                elif other_val is None:
                    val_d[key] = val
                else:
                    val_d[key] = val + other_val
            return PTValues(val_d=val_d)
        else:
            # too much ambiguity about when there are key=None entries
            raise NotImplementedError()

    def __mul__(self, other):
        if isinstance(other, PTValues):
            val_d = {}
            for key, val in self.val_d.items():
                other_val = other.val_d[key]
                if val is None and other_val is None:
                    val_d[key] = None
                elif val is None:
                    val_d[key] = None
                elif other_val is None:
                    val_d[key] = None
                else:
                    val_d[key] = val * other_val
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
