import numpy as np
import pint
from pydantic import BaseModel
from .enums import PT
import matplotlib.pyplot as plt
from . import sts


def pick_v_unit(ot):
    v_units = list(set(ot.v_units()))
    if len(v_units) == 1:
        v_unit, = v_units
    else:
        v_unit = v_units[0]
        for other_v_unit in v_units[1:]:
            try:
                (1 * v_unit).to(other_v_unit)
            except Exception as exc:
                raise ValueError(v_units) from exc
    return v_unit


def scatter(ot, legend_loc=None, title=None, v_unit=None, ignore_vals_above=None, **kwargs):
    t_unit, = set(ot.t_units())
    fig, ax = plt.subplots(1, 1)
    assert ot.ndim == 1 # expected it dims being enums.PT
    v_unit = v_unit or pick_v_unit(ot)
    for val, pt in zip(ot, ot.dims[0]):
        if isinstance(val, pint.Quantity):
            if ignore_vals_above is None or abs(val.magnitude) < ignore_vals_above:
                ax.axhline(val.to(v_unit).magnitude)
        elif val.times:
            times = val.times
            vals = [(vv * val.v_unit).to(v_unit).magnitude for vv in val.values[1:]]
            if ignore_vals_above is None:
                ax.scatter(times, vals, label=pt.two_letter_code(), **kwargs)
            else:
                times_vals = list(zip(*[(tt, vv)
                                    for tt, vv in zip(val.times, val.values[1:])
                                    if abs(vv) < ignore_vals_above]))
                if times_vals:
                    try:
                        times, vals = times_vals
                    except:
                        print(list(times_vals))
                        raise
                    ax.scatter(times, vals, label=pt.two_letter_code(), **kwargs)
        else:
            raise NotImplementedError(val)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(t_unit)
    ax.set_ylabel(v_unit)
    if legend_loc:
        ax.legend(loc=legend_loc)
    return fig, ax


def scatter_subplots(ot, legend_loc=None, v_unit_by_outer_key={}, **kwargs):
    t_unit, = set(ot.t_units())

    if len(ot.dims) > 2:
        raise NotImplementedError()
    if set(ot.dims[1].keys()) != set(PT):
        raise NotImplementedError()
    subplots_args, subplots_kwargs = {
        1: ((1, 1), {'figsize': [5, 4]}),
        2: ((1, 2), {'figsize': [9, 4]}),
        3: ((1, 3), {'figsize': [12, 4]}),
        4: ((2, 2), {'figsize': [8, 7]}),
        5: ((2, 3), {'figsize': [12, 7]}),
        6: ((2, 3), {'figsize': [12, 7]}),
        7: ((2, 4), {'figsize': [14, 7]}),
        8: ((2, 4), {'figsize': [14, 7]}),
        9: ((3, 3), {'figsize': [12, 10]}),
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
        v_unit = v_unit_by_outer_key.get(dim_elem) or pick_v_unit(oti)
        for val, pt in zip(oti, ot.dims[1]):
            if isinstance(val, pint.Quantity):
                if val.magnitude != 0:
                    ax.axhline(val.to(v_unit).magnitude)
            elif val.times:
                ax.scatter(
                    val.times,
                    [(vv * val.v_unit).to(v_unit).magnitude for vv in val.values[1:]],
                    label=pt.two_letter_code(),
                    **kwargs)
        ax.set_title(dim_elem.value)
        ax.set_xlabel(t_unit)
        ax.set_ylabel(v_unit)
        if legend_loc:
            ax.legend(loc=legend_loc)
    fig.tight_layout()
    return fig, axs


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
