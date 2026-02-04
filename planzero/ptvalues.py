import numpy as np
import pint
from pydantic import BaseModel
from .enums import PT
import matplotlib.pyplot as plt
from . import sts

def t_units_of_objtensor(ot):
    t_units = set()
    for obj in ot.ravel():
        try:
            t_units.add(obj.t_unit)
        except (AttributeError, TypeError):
            pass
    return t_units


def v_units_of_objtensor(ot):
    v_units = set()
    for obj in ot.ravel():
        try:
            v_units.add(obj.u)
        except (AttributeError, TypeError):
            try:
                v_units.add(obj.v_unit)
            except (AttributeError, TypeError):
                pass
    return v_units


def scatter(ot, legend_loc=None, **kwargs):
    t_unit, = t_units_of_objtensor(ot)

    if len(ot.dims) > 2:
        raise NotImplementedError()
    if set(ot.dims[1].keys()) != set(PT):
        raise NotImplementedError()
    subplots_args, subplots_kwargs = {
        1: ((1, 1), {'figsize': [4, 4]}),
        2: ((1, 2), {'figsize': [6, 4]}),
        3: ((1, 3), {'figsize': [9, 4]}),
        4: ((2, 2), {'figsize': [8, 7]}),
        5: ((2, 3), {'figsize': [10, 7]}),
        6: ((2, 3), {'figsize': [10, 7]}),
        7: ((2, 4), {'figsize': [14, 7]}),
        8: ((2, 4), {'figsize': [14, 7]}),
        9: ((3, 3), {'figsize': [10, 10]}),
        10: ((3, 4), {'figsize': [14, 10]}),
        11: ((3, 4), {'figsize': [14, 10]}),
        12: ((3, 4), {'figsize': [14, 10]}),
        13: ((4, 4), {'figsize': [14, 13]}),
        14: ((4, 4), {'figsize': [14, 13]}),
        15: ((4, 4), {'figsize': [14, 13]}),
        16: ((4, 4), {'figsize': [14, 13]}),
    }[ot.shape[0]]
    fig, axs = plt.subplots(*subplots_args, **subplots_kwargs)

    for dim_elem, ax, oti in zip(ot.dims[0], axs.flatten(), ot):
        v_unit, = v_units_of_objtensor(oti)
        # guarantees that all pint and SparseTimeSeries have same unit
        for val, pt in zip(oti, ot.dims[1]):
            if isinstance(val, pint.Quantity):
                if val.magnitude != 0:
                    ax.axhline(val.magnitude)
            elif val.times:
                ax.scatter(
                    val.times,
                    val.values[1:],
                    label=pt.value,
                    **kwargs)
        ax.set_title(dim_elem.value)
        ax.set_xlabel(t_unit)
        ax.set_ylabel(v_unit)
        if legend_loc:
            ax.legend(loc=legend_loc)
    fig.tight_layout()


def stacked_bar(ot, legend_loc, **kwargs):
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
