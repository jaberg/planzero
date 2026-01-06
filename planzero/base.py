# TODO:
# For more accurate climate simulation, check out
# https://climate-assessment.readthedocs.io/en/latest/index.html

import bisect
import contextlib
import heapq
from io import StringIO
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pint
import torch
torch.set_default_dtype(torch.float64)

ureg = pint.UnitRegistry()
ureg.define('CAD = [currency]')
ureg.define('people = [human_population]')
ureg.define('cattle = [bovine_population]')

ureg.define('fraction = [] = frac')
ureg.define('ppm = 1e-6 fraction')
ureg.define('ppb = 1e-9 fraction')
u = ureg

# TODO: a global table of official floating-point values of year-start times,
#       for use by annual step functions, accounting math etc.
#       to avoid floating point rounding errors where years are supposed to line up
#       Same for months, maybe weeks.


from . import stakeholders
from . import ipcc_canada

class SparseTimeSeries(object):

    t_unit = getattr(u, 'seconds')

    nan_is_poison = True

    @property
    def v_unit(self):
        return self.values[0].u

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not hasattr(self, '_name'):
            self._name = name
        else:
            assert self._name == name, 'to really change the name, access `_name`'

    def __string__(self):
        return f'STS(name={self.name})'

    def __init__(self, times=None, values=None, unit=None, name=None, t_unit=None, default_value=float('nan')):
        self.times = []
        self._current_readers = []
        self._writer = None
        t_unit = t_unit or self.t_unit
        self.t_unit = t_unit

        if isinstance(unit, str):
            unit = getattr(u, unit)

        if isinstance(default_value, (int, float)):
            if unit is not None:
                self.values = [torch.tensor(default_value) * unit]
            else:
                self.values = [torch.tensor(default_value) * values.u]
        elif hasattr(default_value, 'magnitude'):
            self.values = [torch.tensor(default_value.magnitude) * default_value.u]
        else:
            raise NotImplementedError()

        if times is None and values is None:
            pass
        else:
            if isinstance(times.magnitude, (int, float)):
                self.times = [(torch.tensor(times.magnitude) * times.u).to(self.t_unit)]
            elif isinstance(times.magnitude, np.ndarray):
                self.times = [(torch.tensor(tt) * times.u).to(self.t_unit) for tt in times.magnitude]
            elif isinstance(times.magnitude, torch.Tensor):
                if times.magnitude.ndim == 0:
                    self.times = [times.to(self.t_unit)]
                else:
                    raise NotImplementedError(times.magnitude)
            else:
                raise NotImplementedError(times.magnitude)

            if isinstance(values.magnitude, (int, float)):
                self.values = [
                    (torch.tensor(default_value) * values.u),
                    (torch.tensor(values.magnitude) * values.u),
                ]
            elif isinstance(values.magnitude, np.ndarray):
                if values.magnitude.ndim == 0:
                    self.values = [
                        (torch.tensor(default_value) * values.u),
                        (torch.tensor(values.magnitude) * values.u),
                    ]
                elif values.magnitude.ndim == 1:
                    self.values = [(torch.tensor(default_value) * values.u)]
                    self.values.extend([torch.tensor(vv) * values.u for vv in values.magnitude])
                else:
                    raise NotImplementedError(values.magnitude)
            elif isinstance(values.magnitude, torch.Tensor):
                if values.magnitude.ndim == 0:
                    self.values = [
                        (torch.tensor(default_value) * values.u),
                        (values.to(self.v_unit)),
                    ]
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError(values.magnitude)

        assert len(self.times) + 1 == len(self.values)
        if name is not None:
            self._name = name
        self.default_value = default_value

    @property
    def _times_pint_array(self):
        return torch.stack([tt.magnitude for tt in self.times]) * self.t_unit

    @property
    def _values_pint_array(self):
        return torch.stack([vv.magnitude for vv in self.values]) * self.v_unit

    def latest_val(self, t_query, inclusive):
        index = None
        if self.times:
            if t_query > self.times[-1]:
                index = len(self.values) - 1
            elif t_query == self.times[-1] and inclusive:
                index = len(self.values) - 1
            elif t_query == self.times[-1] and not inclusive:
                index = len(self.values) - 2
            else:
                assert t_query < self.times[-1]
                if inclusive:
                    index = bisect.bisect_right(self.times, t_query)
                else:
                    index = bisect.bisect_left(self.times, t_query)
        else:
            index = 0
        return self.values[index]

    def latest_vals(self, t_query, inclusive=True):

        times = self.times
        values = self.values

        if len(times) == 0:
            indices = torch.tensor([0] * len(t_query))
        else:
            times_array = torch.stack([tt.magnitude for tt in times])
            indices = torch.searchsorted(times_array, t_query.to(self.t_unit).magnitude, side='right')

        rval = torch.stack([values[ii].magnitude for ii in indices])
        if self.nan_is_poison and rval.isnan().any():
            if self.times:
                raise IndexError(f'Series is undefined before time t={self.times[0]}')
            else:
                raise IndexError(f'Series is undefined')
        return rval * self.v_unit

    def append(self, t, v):
        if len(self.times):
            assert t > self.times[-1]

        if isinstance(t.magnitude, (float, int)):
            tt = torch.tensor(t.to(self.t_unit).magnitude) * self.t_unit
        elif isinstance(t.magnitude, torch.Tensor):
            assert t.magnitude.ndim == 0
            tt = t
        else:
            raise NotImplementedError(t)

        if isinstance(v.magnitude, torch.Tensor):
            vv = v.to(self.v_unit)
        elif isinstance(v.magnitude, (float, int)):
            vv = torch.tensor(v.to(self.v_unit).magnitude) * self.v_unit
        else:
            raise NotImplementedError()
        assert tt.u == self.t_unit
        self.times.append(tt)
        self.values.append(vv)

    def extend(self, times, values):
        assert len(times) == len(values)
        if len(times) == 0:
            return

        if self.times:
            assert times[0] > self.times[-1]
        self.times.extend(t.to(self.t_unit))
        self.values.extend(v.to(self.v_unit))

    def plot(self, t_unit=None, annotate=True, **kwargs):
        t_unit = t_unit or self.t_unit
        plt.scatter(
            [t.to(t_unit).magnitude for t in self.times],
            [v.magnitude for v in self.values[1:]],
            **kwargs)
        plt.xlabel(t_unit)
        plt.ylabel(self.v_unit)
        if self.name:
            plt.title(self.name)
        if annotate:
            self.annotate_plot(t_unit=t_unit, **kwargs)

    def annotate_plot(self, t_unit=None, **kwargs):
        """Called once per variable name in comparison plots"""
        pass


class Project(object):

    def __init__(self):
        self._name = None
        self._sub_projects = []

    def init_add_subprojects(self, sub_projects):
        self._sub_projects.extend(sub_projects)

    @property
    def name(self):
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, name):
        self._name = name

    def __string__(self):
        return f'Project(name={self.name})'

    def on_add_project(self, state):
        pass

    def step(self, state):
        return None # return t_next to be called again, None to be left alone

    def project_graph_svg(self, config, state, comparison):
        fig = plt.figure()
        fig.set_layout_engine("constrained")
        key = config['sts_key']
        if config.get('figtype', 'plot') == 'plot':
            sts = state.sts[key]
            sts.plot(t_unit=config.get('t_unit'), label=self.name)
        elif config.get('figtype') == 'plot vs baseline':
            sts = state.sts[key]
            sts.plot(t_unit=config.get('t_unit'), label=self.name)
            comparison.state_B.sts[key].plot(
                t_unit=config.get('t_unit'),
                label='Baseline',
                )
            plt.legend(loc='lower left')
        elif config.get('figtype') == 'plot delta':
            years = comparison._years()
            vals_A = comparison.state_A.sts[key].latest_vals(years, inclusive=True)
            vals_B = comparison.state_B.sts[key].latest_vals(years, inclusive=True)
            diff = vals_A - vals_B
            plt.plot(
                years.to(config['t_unit']).magnitude,
                diff.magnitude)
            plt.title(f'{key} Delta: Active - Inactive')
            plt.xlabel(config['t_unit'])
            plt.ylabel(diff.u)
        else:
            raise NotImplementedError(config.get('figtype'))
        plt.grid()

        svg_buffer = StringIO()
        plt.savefig(svg_buffer, format="svg")
        plt.close()
        svg_string = svg_buffer.getvalue()
        return svg_string


class State(object):
    t_start = 1990 * u.years

    def __init__(self, t_start=t_start, name=None):
        self.t_start = t_start
        self._t_now = torch.tensor(t_start.to(u.seconds).magnitude) * u.seconds
        self.sts = {}
        self.sectoral_emissions_contributors = {
            catpath: {} for catpath in sorted(ipcc_canada.catpaths)}
        self.projects = {}
        self.project_writes = {} # prj-> set of string names
        self.project_requires_current = {} # prj -> set of string names
        self.project_t_next = {} # prj -> t_next
        self._depgraph = None
        self.name = name

    def dependency_digraph(self):
        graph = nx.DiGraph()
        for sts in self.sts.values():
            graph.add_node(sts)
            if sts._writer:
                graph.add_edge(sts._writer, sts)
            for prj in sts._current_readers:
                graph.add_edge(sts, prj)
        for prj in self.projects.values():
            graph.add_node(prj)
        return graph

    @contextlib.contextmanager
    def requiring_latest(self, project):
        class Context(object):
            def __setattr__(_, name, sts):
                if name not in self.sts:
                    self.sts[name] = sts
                    self._depgraph = None
                    sts.name = name
                else:
                    # TODO type-check for compatible existing sts
                    pass
                return self.sts[name]
        try:
            yield Context()
        finally:
            pass

    @contextlib.contextmanager
    def requiring_current(self, project):
        class Context(object):

            def will_read_current(_, name):
                self.sts[name]._current_readers.append(project)
                self.project_requires_current[project.name].add(name)
                self._depgraph = None

            def __setattr__(_, name, sts):
                if name not in self.sts:
                    self.sts[name] = sts
                    sts.name = name
                else:
                    # TODO type-check for compatible existing sts
                    pass
                _.will_read_current(name)
                return self.sts[name]
        try:
            yield Context()
        finally:
            pass

    @contextlib.contextmanager
    def defining(self, project, catpath=None, ghg=None):
        class Context(object):
            def __setattr__(_, name, sts):
                if name in self.sts:
                    # TODO type-check for compatible existing sts
                    pass
                else:
                    self.sts[name] = sts
                    sts.name = name
                assert self.sts[name]._writer is None
                self.sts[name]._writer = project
                self.project_writes[project.name].add(name)
                self._depgraph = None
                return self.sts[name]
        try:
            yield Context()
        finally:
            pass

    def add_project(self, project):
        assert project.name not in self.projects
        self.projects[project.name] = project
        self.project_writes[project.name] = set() # of strings
        self.project_requires_current[project.name] = set() # of strings
        self.project_t_next[project] = project.on_add_project(self)
        self._depgraph = None

        for sub_project in project._sub_projects:
            self.add_project(sub_project)

    def add_projects(self, projects):
        for project in projects:
            self.add_project(project)

    @property
    def t_now(self):
        return self._t_now

    @t_now.setter
    def t_now(self, t_next):
        assert t_next >= self._t_now
        self._t_now = t_next

    def register_emission(self, category_path, ghg, sts_key):
        assert ghg in ('CO2', 'CH4', 'N2O')
        assert sts_key in self.sts
        self.sectoral_emissions_contributors[category_path].setdefault(ghg, []).append(sts_key)

    @property
    def latest(self):
        class Latest(object):
            def __getattr__(_, attr):
                return self.sts[attr].latest_val(self.t_now, inclusive=False)
        return Latest()

    def _current(self, readable_attrs, writeable_attrs):
        readable = set(readable_attrs)
        writeable = set(writeable_attrs)

        class Current(object):

            def __getattr__(_, attr):
                if attr in readable or attr in writeable:
                    # TODO: it might catch errors to be strict about not reading
                    # before writing but current.foo += 1 is such natural syntax
                    return self.sts[attr].latest_val(self.t_now, inclusive=True)
                else:
                    if attr in self.sts:
                        raise AttributeError(f'current value of attribute {attr} not available')
                    else:
                        raise AttributeError(attr)

            def __setattr__(_, attr, val):
                if attr in writeable:
                    self.sts[attr].append(self.t_now, val)
                    readable.add(attr)
                    writeable.remove(attr)
                else:
                    assert 0

        return Current()

    def plot(self, t_unit='years', **kwargs):
        if len(self.sts) <= 1:
            fig = plt.figure()
            rows = 1
            cols = 1
        elif len(self.sts) <= 6:
            fig = plt.figure(figsize=[12, 8])
            rows = 2
            cols = 3
        elif len(self.sts) <= 15:
            rows = 5
            cols = 3
            fig = plt.figure(figsize=[12, (len(self.sts) // cols + 1) * 5.5])
        else:
            raise NotImplementedError()
        fig.set_layout_engine("constrained")

        for ii, sts in enumerate(self.sts.values()):
            plt.subplot(rows, cols, ii + 1)
            sts.plot(t_unit=t_unit)

    def run_until(self, t_stop):
        if self._depgraph is None:
            self._depgraph = self.dependency_digraph()
            self._heap = [
                (self.project_t_next[prj], ii, prj)
                for (ii, prj) in enumerate(nx.topological_sort(self._depgraph))
                if isinstance(prj, Project) and self.project_t_next[prj] is not None]
            heapq.heapify(self._heap)

        while self.t_now <= t_stop and self._heap:
            t_next, node_idx, prj = heapq.heappop(self._heap)
            assert t_next >= self.t_now
            self.t_now = t_next

            new_t_next = prj.step(
                self,
                current=self._current(
                    readable_attrs=self.project_requires_current[prj.name],
                    writeable_attrs=self.project_writes[prj.name]))
            self.project_t_next[prj] = new_t_next
            if new_t_next is not None:
                assert new_t_next > self.t_now
                heapq.heappush(self._heap, (new_t_next, node_idx, prj))

class GlobalHeatEnergy(SparseTimeSeries):
    pass

    #def annotate_plot(self, t_unit=None, **kwargs):
        #height = specific_heat_of_top_300m_of_ocean.to(self.v_unit / u.kelvin).magnitude
        #plt.axhline(height)
        #plt.text(self.times[0].to('years').magnitude, height, "Shallow Ocean 1C")


class AtmosphericChemistry(Project):
    methane_decay_timescale = 10.0
    methane_GWP = 28.0 # global warming potential, for CO2e calculation
    molar_mass_CH4 = 16.0
    molar_mass_CO2 = 44.0

    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            for catpath, contributors in state.sectoral_emissions_contributors.items():
                for sts_key in contributors.get('CO2', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('CH4', []):
                    ctx.will_read_current(sts_key)

        with state.defining(self) as ctx:
            for catpath, _ in state.sectoral_emissions_contributors.items():
                setattr(ctx, f'Predicted_Annual_Emitted_CO2_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_CH4_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_CO2e_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))

            ctx.Predicted_Annual_Emitted_CO2_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_CH4_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_CO2e_mass = SparseTimeSeries(unit=u.kiloton)

            ctx.Atmospheric_CO2_conc = SparseTimeSeries(unit=u.ppm, default_value=400.0 * u.ppm)
            ctx.Atmospheric_CH4_conc = SparseTimeSeries(unit=u.ppb, default_value=1775.0 * u.ppb)

            ctx.DeltaF_CO2 = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_CH4 = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_forcing = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_feedback = SparseTimeSeries(unit=u.petawatt)

            ctx.Heat_Energy_forcing = SparseTimeSeries(unit=u.exajoule)
            ctx.Heat_Energy_imbalance = SparseTimeSeries(unit=u.exajoule)
            ctx.Cumulative_Heat_Energy = GlobalHeatEnergy(default_value=0.0 * u.exajoule)
            ctx.Ocean_Temperature_Anomaly = SparseTimeSeries(default_value=1.3 * u.kelvin)

        return state.t_now

    def step(self, state, current):
        # add up annual emissions from registry
        annual_CO2_mass = ureg.Quantity("0 kiloton")
        annual_CH4_mass = ureg.Quantity("0 kiloton")
        for catpath, contributors in state.sectoral_emissions_contributors.items():
            catpath_CO2_mass = 0 * u.kiloton

            for sts_key in contributors.get('CO2', []):
                catpath_CO2_mass += getattr(current, sts_key)

            catpath_CH4_mass = 0 * u.kiloton
            for sts_key in contributors.get('CH4', []):
                catpath_CH4_mass += getattr(current, sts_key)

            setattr(current, f'Predicted_Annual_Emitted_CO2_mass_{catpath}', catpath_CO2_mass)
            setattr(current, f'Predicted_Annual_Emitted_CH4_mass_{catpath}', catpath_CH4_mass)
            setattr(current, f'Predicted_Annual_Emitted_CO2e_mass_{catpath}',
                catpath_CO2_mass
                + catpath_CH4_mass * self.methane_GWP)

            annual_CO2_mass += catpath_CO2_mass
            annual_CH4_mass += catpath_CH4_mass

        current.Predicted_Annual_Emitted_CO2_mass = annual_CO2_mass
        current.Predicted_Annual_Emitted_CH4_mass = annual_CH4_mass
        current.Predicted_Annual_Emitted_CO2e_mass = annual_CO2_mass + self.methane_GWP * annual_CH4_mass

        # apply an atmospheric climate model
        annual_CO2_mass_atmospheric = annual_CO2_mass * .45 # Gemini claimed a lot gets absorbed by sinks, keep this much
        annual_CH4_mass_atmospheric = annual_CH4_mass * 1.0 # no such discounting of CH4

        annual_emitted_CO2_in_atmosphere_as_concentration = (
            annual_CO2_mass_atmospheric
            / (7.8 * u.gigatonne / u.ppm)) # amount required to raise concentration by 1 ppm

        annual_emitted_CH4_in_atmosphere_as_concentration = (
            annual_CH4_mass_atmospheric
            / (2.78 * u.megatonne / u.ppb))

        tau_ch4 = 10.0 # about 10 years
        annual_ch4_to_co2_decay = (
            state.latest.Atmospheric_CH4_conc
            / tau_ch4)

        current.Atmospheric_CH4_conc += (
            annual_emitted_CH4_in_atmosphere_as_concentration
            + 180 * u.ppb # baseline from other sources
            - annual_ch4_to_co2_decay)

        current.Atmospheric_CO2_conc += (
            annual_emitted_CO2_in_atmosphere_as_concentration
            + 2 * u.ppm # baseline from other sources
            + annual_ch4_to_co2_decay
            - (current.Atmospheric_CO2_conc / 200) # decay timescale for CO2?
        )

        surface_area_of_earth = 5.1e14 * u.m * u.m

        reference_CO2_conc = torch.tensor(280.0) * u.ppm
        current.DeltaF_CO2 = (
            5.35 * u.watt / (u.m * u.m)
            * surface_area_of_earth
            * torch.log(current.Atmospheric_CO2_conc.to(u.ppm).magnitude
                        / reference_CO2_conc.to(u.ppm).magnitude))

        reference_CH4_conc = torch.tensor(722.0) * u.ppb
        current.DeltaF_CH4 = (
            0.036 * u.watt / (u.m * u.m)
            * surface_area_of_earth
            * (torch.sqrt(current.Atmospheric_CH4_conc.to(u.ppb).magnitude)
               - torch.sqrt(reference_CH4_conc.to(u.ppb).magnitude)))
        # TODO: include NO2 overlap correction term

        current.DeltaF_forcing = (
            current.DeltaF_CO2
            + current.DeltaF_CH4)

        current.DeltaF_feedback = (
            -1.3 * u.watt / (u.m * u.m) / u.kelvin
            * surface_area_of_earth
            * current.Ocean_Temperature_Anomaly) # will be stale value

        current.Heat_Energy_forcing = (
            self.stepsize # integrate over duration of stepsize aka 1 year
            * current.DeltaF_forcing)

        current.Heat_Energy_imbalance = (
            self.stepsize # integrate over duration of stepsize aka 1 year
            * (current.DeltaF_forcing + current.DeltaF_feedback))

        specific_heat_of_top_200m_of_ocean = 151200.0 * u.exajoule / u.kelvin * 2
        current.Ocean_Temperature_Anomaly += (
            current.Heat_Energy_imbalance
            / (specific_heat_of_top_200m_of_ocean))

        current.Cumulative_Heat_Energy += current.Heat_Energy_imbalance

        return state.t_now + self.stepsize


class GeometricHumanPopulationForecast(Project):
    def __init__(self, rate=1.014):
        super().__init__()
        self.rate = rate
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        assert 1989 * u.years <= state.t_now <= 1991 * u.years, state.t_now.to(u.years)

        with state.defining(self) as ctx:
            ctx.human_population = SparseTimeSeries(state.t_now, 27_685_730 * u.people)

        return state.t_now + self.stepsize

    def step(self, state, current):
        current.human_population *= self.rate
        return state.t_now + self.stepsize


class GeometricBovinePopulationForecast(Project):

    # 70% of emissions remain, according to https://www.helsinki.fi/en/news/climate-change/new-feed-additive-can-significantly-reduce-methane-emissions-generated-ruminants-already-dairy-farm
    # https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html
    bovaer_methane_reduction_fraction = .625
    # TODO: dairy cattle average is .7
    # TODO: beef cattle reduction is greater, multiplier should be .55

    methane_per_head_per_year = 175 * u.pounds / u.cattle
    # https://www.ucdavis.edu/food/news/making-cattle-more-sustainable

    def __init__(self):
        super().__init__()
        csv = pd.read_csv(os.path.join(os.environ['PLANZERO_DATA'], 'number-of-cattle.csv'))
        self.jan1 = csv[(csv['Farm type'] == 'On all cattle operations') & (csv['Survey date'] == 'At January 1')]
        self.jul1 = csv[(csv['Farm type'] == 'On all cattle operations') & (csv['Survey date'] == 'At July 1')]
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = SparseTimeSeries(
                default_value=0 * u.dimensionless)
        with state.defining(self) as ctx:
            ctx.bovine_population_on_bovaer = SparseTimeSeries(
                default_value=0 * u.cattle)
            ctx.bovine_population = SparseTimeSeries(
                self.jan1['REF_DATE'].values * u.years,
                (.5 * self.jan1['VALUE'].values * 1000
                 + .5 * self.jul1['VALUE'].values * 1000) * u.cattle)
            ctx.bovine_methane = SparseTimeSeries(
                state.t_now,
                (state.sts['bovine_population'].latest_val(2000 * u.years, inclusive=True)
                 * self.methane_per_head_per_year))
        state.register_emission('Enteric_Fermentation', 'CH4', 'bovine_methane')
        return 2000 * u.years + self.stepsize

    def step(self, state, current):
        if state.t_now >= 2026 * u.years:
            # we've run out of data at this point, start guessing
            current.bovine_population = (
                # .995 kind of looks smoother, but constant is more defensible?
                max(
                    12_500_000 * .999 ** (state.t_now.to('years').magnitude - 2010),
                    2_000_000
                   ) * u.cattle)

        current.bovine_population_on_bovaer = (
            current.bovine_population
            * current.bovine_population_fraction_on_bovaer)

        current.bovine_methane = (
            (current.bovine_population_on_bovaer
             * self.methane_per_head_per_year
             * self.bovaer_methane_reduction_fraction)
            + (
                (current.bovine_population - current.bovine_population_on_bovaer)
                * self.methane_per_head_per_year))

        return state.t_now + self.stepsize


class NationalBovaerMandate(Project):
    after_tax_cashflow_name = f'NationalBovaerMandate_AfterTaxCashflow'

    bovaer_price = 150 * u.CAD / u.cattle / u.year

    def __init__(self, idea=None, peak_year=2035 * u.year, shoulder_years=5 * u.years):
        super().__init__()
        self.idea = idea
        self.peak_year = peak_year
        self.shoulder_years = shoulder_years
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_latest(self) as ctx:
            ctx.bovine_population_on_bovaer = SparseTimeSeries(
                default_value=0 * u.cattle)
        with state.defining(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    def step(self, state, current):
        zval = (
            (state.t_now - self.peak_year)
            / self.shoulder_years)
        current.bovine_population_fraction_on_bovaer = torch.sigmoid(
            zval.to('dimensionless').magnitude) * u.dimensionless
            
        setattr(
            current,
            self.after_tax_cashflow_name, 
            -state.latest.bovine_population_on_bovaer
            * self.bovaer_price * self.stepsize)

        return state.t_now + self.stepsize

    def project_page_vision(self): 
        return f"""
        <a href="https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html">Bovaer</a>
        is a feed supplement that reduces the methane produced by bovine digestion.
        A national Bovaer mandate would phase in the use of Bovaer nation-wide for all cattle.
        There is no obvious economic benefit (or harm) to farmers for using Bovaer, so
        it would be appropriate for a governing body to pay for the additive and drive adoption through regulation.
        """

    def project_page_graphs(self):
        rval = []

        descr = f"""
        This analysis uses the following place-holder projection of the national bovine herd size, that extrapolates a very gradual decline from the current size.
        """
        rval.append(dict(
            sts_key='bovine_population',
            t_unit='years',
            descr=descr))

        descr = f"""
        This project supposes a sigmoidal adoption curve of Bovaer, centered at year {self.peak_year.to('years').magnitude}.
        """
        rval.append(dict(
            sts_key='bovine_population_fraction_on_bovaer',
            t_unit='years',
            descr=descr))


        descr = """
        We therefore see a dropping annual emissions of bovine methane, partly through the adoption of Bovaer, and partly due to the gradual reduction in population.
        """
        rval.append(dict(
            sts_key='bovine_methane',
            t_unit='years',
            descr=descr))

        descr = """
        As Bovaer is adopted, the impact on Canada's emissions is modulated by the number of cattle.
        The size of Canada's national herd has declined over the last decade, the future is not known.
        """
        rval.append(dict(
            sts_key='bovine_population_on_bovaer',
            t_unit='years',
            descr=descr))

        descr = """
        The impact on national methane emissions from so-called enteric fermentation is expected to be significant.
        """
        rval.append(dict(
            sts_key='Predicted_Annual_Emitted_CH4_mass',
            t_unit='years',
            figtype='plot vs baseline',
            descr=descr))

        descr = """
        The impact on global atmospheric methane concentration is expected to be noticeable.
        """
        rval.append(dict(
            sts_key='Atmospheric_CH4_conc',
            t_unit='years',
            figtype='plot vs baseline',
            descr=descr))

        descr = """
        The impact on global heat forcing is too small to see on a graph of that phenomenon.
        """
        rval.append(dict(
            sts_key='DeltaF_forcing',
            t_unit='years',
            figtype='plot vs baseline',
            descr=descr))

        descr = """
        Still, a visualization of the difference in global heat forcing reveals the shape of the impact over time.
        The datapoints in this curve are used to compute the Net Present Heat for the project, by adding up the energy associated with each year (modulated by the future discount factor).
        """
        rval.append(dict(
            sts_key='DeltaF_forcing',
            t_unit='years',
            figtype='plot delta',
            descr=descr))

        descr = """
        In other terms, the difference in global heat forcing due to a national Bovaer mandate can be quantified as a small change in (upward) temperature trajectory for the top 200m of the world's oceans.
        """
        rval.append(dict(
            sts_key='Ocean_Temperature_Anomaly',
            t_unit='years',
            figtype='plot delta',
            descr=descr))

        descr = f"""
        In terms of financial modelling, the project assumes a price of Bovaer ( {NationalBovaerMandate.bovaer_price} ) that remains constant for the next 200 years. At the scale of production associated with national adoption in Canada and in other countries, this is arguably an over-estimate.
        The curve is simply the product of the population on Bovaer with the price per head.
        The datapoints in this curve are used to compute the Net Present Value for the project.
        """
        rval.append(dict(
            sts_key='NationalBovaerMandate_AfterTaxCashflow',
            t_unit='years',
            figtype='plot',
            descr=descr))

        return rval



class ProjectComparison(object):
    def __init__(self, state_A, state_B, present, project):
        self.state_A = state_A # state with project
        self.state_B = state_B # baseline state
        self.present = present
        self.project = project

    def _years(self):
        t_start = min(self.state_A.t_start, self.state_B.t_start)
        t_stop = max(self.state_A.t_now, self.state_B.t_now)
        start_year = int(t_start.to('years').magnitude)
        stop_year = int(t_stop.to('years').magnitude) + 1
        years = torch.arange(start_year, stop_year) * u.years
        return years

    def years_as_list(self):
        return [int(year.to('years').magnitude) for year in self._years()]

    @property
    def _present_year_int(self):
        return int(self.present.to('years').magnitude)

    def _net_present_envelope(self, years, base_rate):
        present_year_int = self._present_year_int
        envelope = [0] * len(years)
        for ii, year in enumerate(years):
            year_int = int(year.to('years').magnitude)
            if year_int >= present_year_int:
                envelope[ii] = base_rate ** (year_int - present_year_int)
        return torch.tensor(envelope)

    def net_present_discounted_sum(self, base_rate, key, inclusive=True):
        years = self._years()
        vals_A = self.state_A.sts[key].latest_vals(years, inclusive=inclusive)
        vals_B = self.state_B.sts[key].latest_vals(years, inclusive=inclusive)
        diff = vals_A - vals_B
        envelope = self._net_present_envelope(years, base_rate)
        return torch.cumsum(diff.magnitude * envelope, dim=0)[-1] * diff.u

    def net_present_CO2e(self, base_rate):
        return self.net_present_discounted_sum(
            base_rate,
            key='Predicted_Annual_Emitted_CO2e_mass')

    def net_present_heat(self, base_rate):
        return self.net_present_discounted_sum(
            base_rate,
            key='Heat_Energy_forcing')

    def net_present_value(self, base_rate):
        if self.project.after_tax_cashflow_name in self.state_B.sts:
            raise NotImplementedError()
        years = self._years()
        envelope = self._net_present_envelope(years, base_rate)
        cashflow = self.state_A.sts[self.project.after_tax_cashflow_name].latest_vals(
            years, inclusive=True)
        return torch.cumsum(cashflow.magnitude * envelope, dim=0)[-1] * cashflow.u

    def cost_per_ton_CO2e(self, base_rate):
        npv = self.net_present_value(base_rate=base_rate)
        npc = self.net_present_CO2e(base_rate=base_rate)
        if npc >= 0:
            return float('nan') * u.CAD / u.tonne
        return (npv / npc).to(u.CAD / u.tonne)

    def echart_series_Mt(self, A_or_B, catpath, stack=None, name=None):
        years = self._years()
        if A_or_B == "A":
            state = self.state_A
        elif A_or_B == "B":
            state = self.state_B
        else:
            raise NotImplementedError(A_or_B)
        predictions = state.sts[f'Predicted_Annual_Emitted_CO2e_mass_{catpath}'].latest_vals(
            years, inclusive=True)
        data = [{'value': float(datum.to(u.megatonne).magnitude),
                 'url': f'/ipcc-sectors/{catpath}'.replace(' ', '_')}
                for datum in predictions]
        rval = dict(
            name=name or catpath,
            type='line',
            #areaStyle={},
            #emphasis={'focus': 'series'},
            data=data)
        if stack:
            rval['stack'] = stack
        return rval


class ProjectEvaluation(object):
    def __init__(self, projects, common_projects, alt_project=None, present=None):
        self.projects = projects # dict
        self.common_projects = common_projects
        self.present = (
            time.time() * u.seconds + 1970 * u.years
            if present is None else present)

        self.comparisons = {}
        self.states = {}
        default_state = None
        for eval_name, prj in projects.items():
            if isinstance(prj, (list, tuple)):
                raise NotImplementedError()
            else:
                state_A = State(name=f'StateA_{eval_name}')
                state_A.add_project(prj)
                state_A.add_projects(common_projects)
                if default_state is None:
                    default_state = State(name=f'Baseline')
                    default_state.add_projects(common_projects)
                self.comparisons[eval_name] = ProjectComparison(
                    state_A=state_A,
                    state_B=default_state,
                    present=self.present,
                    project=prj)
                self.states[state_A.name] = state_A
                self.states[default_state.name] = default_state

    def run_until(self, t_stop):
        for state in self.states.values():
            state.run_until(t_stop)

    def all_sts_names(self):
        rval = set()
        for state in self.states.values():
            rval.update(state.sts.keys())
        return rval

    def plot(self, t_unit='years', **kwargs):

        sorted_sts_names = list(sorted(self.all_sts_names()))

        if len(sorted_sts_names) <= 1:
            fig = plt.figure()
            rows = 1
            cols = 1
        elif len(sorted_sts_names) <= 6:
            fig = plt.figure(figsize=[12, 8])
            rows = 2
            cols = 3
        elif len(sorted_sts_names) <= 15:
            rows = 5
            cols = 3
            fig = plt.figure(figsize=[12, (len(sorted_sts_names) // cols + 1) * 5.5])
        elif len(sorted_sts_names) <= 28:
            rows = 7
            cols = 4
            fig = plt.figure(figsize=[15, (len(sorted_sts_names) // cols + 1) * 5.5])
        else:
            raise NotImplementedError()
        fig.set_layout_engine("constrained")

        for ii, sts_name in enumerate(sorted_sts_names):
            plt.subplot(rows, cols, ii + 1)
            for state in self.states.values():
                if sts_name in state.sts:
                    state.sts[sts_name].plot(t_unit=t_unit, annotate=True, label=state.name)

    def plot_nph_vs_npv(self, discount_rate, nph_unit='exajoule', npv_unit='MCAD'):
        base_rate = (1 - discount_rate)
        eval_names = []
        nph1s = []
        npv1s = []
        for eval_name, cmp in self.comparisons.items():
            nph1 = cmp.net_present_heat(base_rate=base_rate)
            npv1 = cmp.net_present_value(base_rate=base_rate)
            eval_names.append(eval_name)
            nph1s.append(nph1.to(nph_unit).magnitude)
            npv1s.append(npv1.to(npv_unit).magnitude)

        plt.figure()
        plt.title(f'Future-Discounted Project Comparison @ {discount_rate * 100:.1f}% ({base_rate ** 100:.2f} at 100 years)')
        plt.scatter(npv1s, nph1s)
        for ii, eval_name in enumerate(eval_names):
            plt.annotate(eval_name, (npv1s[ii], nph1s[ii]))
        plt.xlabel(f'Net Present Value ({npv_unit})')
        plt.ylabel(f'Net Present Heat Forcing ({nph_unit})')

    def iter_npv_nph_evalname(self, discount_rate):
        base_rate = (1 - discount_rate)
        for eval_name, cmp in self.comparisons.items():
            nph = cmp.net_present_heat(base_rate=base_rate)
            npv = cmp.net_present_value(base_rate=base_rate)
            yield npv, nph, eval_name


# carbon capture
# negative-carbon cement
# enhanced rock weathering (e.g. UNDO, Carbon Run)
# Buying an EV
# Steel production
# Fertilizer production
# Farm machinery
# on-farm biogas capture
# rooftop solar
# highway umbrella solar
# photovoltaic sail boats for cargo
# tethered balloon heat sinks, wind turbines, and solar farms, and transportation medium
# what about India's cattle population!?
# new process for hydrogen peroxide: https://interestingengineering.com/innovation/solar-hydrogen-peroxide-cornell-breakthrough

class ComboA(Project):
    def __init__(self, idea):
        super().__init__()
        self.idea = idea
        self.init_add_subprojects([
            NationalBovaerMandate(idea=stakeholders.ideas.national_bovaer_mandate),
            BatteryTugWithAuxSolarBarges(idea=stakeholders.ideas.battery_tugs_w_aux_solar_barges),
            Force_Government_ZEVs(),
        ])
        self.after_tax_cashflow_name = f'{self.__class__.__name__}_AfterTaxCashFlow'
        self.stepsize = 1.0 * u.years

    def project_page_vision(self):
        return ""

    def project_page_intro(self):
        rval = "<p>Projects included:"
        rval += "<ul>"
        for proj in self._sub_projects:
            rval += f"<li><a href='/strategies/{proj.idea.name}/'>{proj.idea.full_name}</a></li>"
        rval += "</ul>"
        rval += "</p>"
        return rval

    def project_page_graphs(self):
        return []

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            for proj in self._sub_projects:
                setattr(ctx, proj.after_tax_cashflow_name, SparseTimeSeries(
                    default_value=0 * u.MCAD))
        with state.defining(self) as ctx:
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    def step(self, state, current):
        cashflow = 0 * u.MCAD
        for proj in self._sub_projects:
            cashflow += getattr(current, proj.after_tax_cashflow_name)
        setattr(current, self.after_tax_cashflow_name, cashflow)
        return state.t_now + self.stepsize


class IPCC_Forest_Land_Model(Project):
    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:

            ctx.Other_Forest_Land_CO2 = SparseTimeSeries(
                default_value=40.0 * u.Mt)

        state.register_emission('Forest_Land', 'CO2', 'Other_Forest_Land_CO2')


class IPCC_Transport_Marine_DomesticNavigation_Model(Project):
    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:

            # PacificLogBargeForecast
            ctx.pacific_log_barge_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

            ctx.Lakers_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

            ctx.Other_Domestic_Navigation_CO2 = SparseTimeSeries(
                default_value=3.4 * u.Mt)

        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2', 'Lakers_CO2')
        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2', 'Other_Domestic_Navigation_CO2')


class IPCC_Transport_RoadTransportation_LightDutyGasolineTrucks(Project):
    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.human_population = SparseTimeSeries(state.t_now, 27_685_730 * u.people)
            ctx.Government_LightDutyGasolineTrucks_ZEV_fraction = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            ctx.Other_LightDutyGasolineTrucks_ZEV_fraction = SparseTimeSeries(
                default_value=0 * u.dimensionless)

        with state.defining(self) as ctx:
            # https://www.canada.ca/en/treasury-board-secretariat/services/innovation/greening-government/government-canada-greenhouse-gas-emissions-inventory.html
            # not exactly using ^^ but guessing based on that and Gemini estimation

            # TODO: factor in CO2e footprint of manufacturing each type of vehicle

            # TODO: factor in the emissions of electricity generation in each province
            #       and the population of each province

            ctx.Government_LightDutyGasolineTrucks_CO2 = SparseTimeSeries(
                default_value=0 * u.Mt)
            ctx.Other_LightDutyGasolineTrucks_CO2 = SparseTimeSeries(
                default_value=0 * u.Mt)

        state.register_emission('Transport/Road_Transportation/Light-Duty_Gasoline_Trucks', 'CO2', 'Other_LightDutyGasolineTrucks_CO2')
        state.register_emission('Transport/Road_Transportation/Light-Duty_Gasoline_Trucks', 'CO2', 'Government_LightDutyGasolineTrucks_CO2')
        return state.t_now + self.stepsize

    def step(self, state, current):
        coefficient = 1_200 * u.kg / u.people
        current.Government_LightDutyGasolineTrucks_CO2 = (
            current.human_population * coefficient * .025
            * (1 * u.dimensionless - current.Government_LightDutyGasolineTrucks_ZEV_fraction))
        current.Other_LightDutyGasolineTrucks_CO2 = (
            current.human_population * coefficient * .975
            * (1 * u.dimensionless - current.Other_LightDutyGasolineTrucks_ZEV_fraction))
        return state.t_now + self.stepsize


class Force_Government_ZEVs(Project):
    idea = stakeholders.ideas.force_government_fleet_to_go_green

    def __init__(self, start_time=2022 * u.years, end_time=2035 * u.years):
        super().__init__()
        self.stepsize = 1.0 * u.years
        self.start_time = start_time
        self.end_time = end_time
        self.after_tax_cashflow_name = f'{self.__class__.__name__}_AfterTaxCashFlow'

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.Government_LightDutyGasolineTrucks_ZEV_fraction = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    def step(self, state, current):
        if state.t_now >= self.start_time:
            ramp_duration = self.end_time - self.start_time
            ramp_time = state.t_now - self.start_time
            if ramp_time >= ramp_duration:
                current.Government_LightDutyGasolineTrucks_ZEV_fraction = 1 * u.dimensionless
            else:
                current.Government_LightDutyGasolineTrucks_ZEV_fraction = (
                    ramp_time / ramp_duration * u.dimensionless)
        # assume that on average, the total cost of ownership nets out about 0
        # which based on work with Roger Martin seems plausible
        # do not assume batteries get cheaper, or vehicles get cheaper
        setattr(
            current,
            self.after_tax_cashflow_name, 
            0 * u.CAD)
        return state.t_now + self.stepsize

    def project_page_vision(self):
        return """
        The technology of ZEVs and PHEVs is maturing, the municipal and civil services of the country are 
        already evaluating these vehicle types extensively, and with some success I believe.
        It would be meaningful for our governments to push the transition to this new technology with their significant scale of operations.
        """

    def project_page_graphs(self):
        rval = []

        descr = f"""
        Input assumption: the shape of the fraction of vehicle roles transitioning to ZEVs.
        """
        rval.append(dict(
            sts_key='Government_LightDutyGasolineTrucks_ZEV_fraction',
            t_unit='years',
            descr=descr))

        descr = f"""
        Estimated CO2 emissions from government light-duty gasoline trucks (TODO: include grid emissions, and at least note [externalized] manufacturing emissions).
        """
        rval.append(dict(
            sts_key='Government_LightDutyGasolineTrucks_CO2',
            t_unit='years',
            descr=descr))

        return rval

class PacificLogBargeForecast(Project):
    # somehow link/merge with stakeholders.PacificLogBarges

    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.n_pacific_log_tugs_ZEV_constructed = SparseTimeSeries(
                default_value=0 * u.dimensionless)
        with state.defining(self) as ctx:
            ctx.n_pacific_log_tugs = SparseTimeSeries(
                default_value=stakeholders.ac.Pacific_Log_Barges.n_barges * u.dimensionless)
            ctx.n_pacific_log_tugs_diesel = SparseTimeSeries(
                default_value=stakeholders.ac.Pacific_Log_Barges.n_barges * u.dimensionless)
            ctx.n_pacific_log_tugs_ZEV = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            ctx.pacific_log_barge_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2',
                                'pacific_log_barge_CO2')
        return state.t_now

    def step(self, state, current):
        current.n_pacific_log_tugs_ZEV = current.n_pacific_log_tugs_ZEV_constructed
        current.n_pacific_log_tugs_diesel = (
            current.n_pacific_log_tugs
            - current.n_pacific_log_tugs_ZEV)
        current.n_pacific_log_tugs = (
            current.n_pacific_log_tugs_diesel
            + current.n_pacific_log_tugs_ZEV)
        current.pacific_log_barge_CO2 = (
            2.68 * u.kg / u.liter
            * 400 * u.liter / u.hour
            * current.n_pacific_log_tugs_diesel
            * (300 / 365) # working most days
            * self.stepsize)

        return state.t_now + self.stepsize


class BatteryTugWithAuxSolarBarges(Project):
    #after_tax_cashflow_name = f'BatteryTugWithAuxSolarBarges_AfterTaxCashflow'
    vancouver_winter_solar_hours_per_day = 1.3 * u.kilowatt * u.hours / u.m ** 2 / u.day

    PV_cost_per_area = 70 * u.CAD / u.m ** 2
    battery_cost_per_capacity = 200 * u.CAD / (1280 * u.watt * u.hour)

    def __init__(self, idea=None, year_0=2026 * u.years):
        super().__init__()
        self.idea = idea
        self.stepsize = 1.0 * u.years
        self.after_tax_cashflow_name = f'{self.__class__.__name__}_AfterTaxCashFlow'
        self.year_0 = year_0
        self.vessel_lifetime = 20 * u.years
        self.r_and_d_duration = 5 * u.years
        self.battery_range_time = 24 * u.hours * 1

        tug_power_required = 2000 * u.horsepower
        if 0:
            # guessing here:
            # this web application: https://nrcan-rncan.maps.arcgis.com/apps/webappviewer/index.html?id=0de6c7c412ca4f6cbd399efedafa4af4&_gl=1*1096veb*_ga*MTg0MDQ2OTMxOS4xNzY0MDE3NDI0*_ga_C2N57Y7DX5*czE3NjUzMTYyMjQkbzMkZzEkdDE3NjUzMTYzMjIkajYwJGwwJGgw
            # estimates north Vancouver Island gets just .7 kwh/m2/day in winter for horizontal panels
            # but maybe 2kwh/m2/day for south-facing vertically-mounted ones
            # if we use a double-sided PV sail, then it will get various amounts of light
            # depending on which way the wind is blowing and which way the vessel is going
            # so...

            # suppose auxiliary ships 50m long with 50m sails, 20m wide
            aux_vessel_length = 70 * u.m
            aux_vessel_beam = 20 * u.m
            pv_area_per_aux_vessel = aux_vessel_length * aux_vessel_beam

            # suppose wind provides no propulsion
            wind_power_fraction = 0.5

            pv_power_required = tug_power_required * (1 - wind_power_fraction)

            self.n_aux_vessels_required = int((
                pv_power_required
                / (pv_area_per_aux_vessel * self.vancouver_winter_solar_hours_per_day)
            ).to('dimensionless') + 1)

            # guessing
            battery_capacity_per_aux_vessel = 0.01 * battery_capacity_per_tug

            # super-guess that the launched cost of pv & battery is half the cost of the vessel
            self.cost_per_aux_vessel = 2 * (
                pv_area_per_aux_vessel * self.PV_cost_per_area
                + battery_capacity_per_aux_vessel * self.battery_cost_per_capacity
            ).to(u.CAD)

        else:
            wind_power_fraction = 0.0

        self.battery_capacity_per_tug = tug_power_required * (1 - wind_power_fraction) * self.battery_range_time

        self.battery_cost_per_tug = (self.battery_capacity_per_tug * self.battery_cost_per_capacity).to(u.CAD)
        self.non_battery_cost_per_tug = 5_000_000 * u.CAD # guess

        self.cost_per_tug = self.battery_cost_per_tug + self.non_battery_cost_per_tug

        self.r_and_d_annual_cost = 1_000_000 * u.CAD / u.year


    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.n_pacific_log_tugs_ZEV_constructed = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now


    @property
    def fuel_cost_rate(self):
        price_of_diesel = 1.20  * u.CAD / u.liter
        working_days_per_year = 300
        diesel_tug_fuel_consumption = 400 * u.liter / u.hour
        # suppose everything same but fuel, so revenue is fuel savings
        fuel_cost_rate = (
            price_of_diesel
            * diesel_tug_fuel_consumption
            * (working_days_per_year / 365)).to(u.CAD/u.year)

        price_of_electricity = 0.1 * u.CAD / (u.kilowatt * u.hour)
        electricity_cost_rate = (
            price_of_electricity
            * self.battery_capacity_per_tug / u.day
            * (working_days_per_year / 365)).to(u.CAD/u.year)
        return fuel_cost_rate - electricity_cost_rate

    @property
    def tug_labour_rate(self):
        crew_size = 9
        avg_salary = 75_000 * u.CAD / u.year
        cost_rate = crew_size * avg_salary
        return cost_rate

    def step(self, state, current):
        manufacturing_costs = 0 * u.CAD
        r_and_d_costs = 0 * u.CAD
        if state.t_now >= self.year_0:
            r_and_d_costs += self.r_and_d_annual_cost * self.stepsize

        if state.t_now >= self.year_0 + self.r_and_d_duration:
            n_tugs_built_per_step = (
                state.latest.n_pacific_log_tugs
                / self.vessel_lifetime) * self.stepsize
            manufacturing_costs += n_tugs_built_per_step * (
                self.cost_per_tug)
            if state.latest.n_pacific_log_tugs_ZEV_constructed < state.latest.n_pacific_log_tugs:
                current.n_pacific_log_tugs_ZEV_constructed += n_tugs_built_per_step
            else:
                # we don't model the explicit decomissioning and recommisioning every 20 years
                pass

        revenue = state.latest.n_pacific_log_tugs_ZEV * self.fuel_cost_rate * self.stepsize
        
        # not included:
        # salvage value, part-reuse between vessel generations
        # maintenance
        # insurance
        # financing

        setattr(
            current,
            self.after_tax_cashflow_name, 
            - r_and_d_costs
            - manufacturing_costs
            + revenue
            )
        return state.t_now + self.stepsize

    def project_page_vision(self): 
        return f"""
        The idea: use battery-powered tugs to move bulk cargo barges.
        Provide range by using grid power, or by constructing solar-farm piers or floating arrays at which tugs can recharge or swap TEU-sized batteries.
        The current financing and environmental impact model is based on a plan of
        simply replacing the fleet of diesel
        tugs pulling log barges around Vancouver Island, and not the Great Lakes or northern supply routes (although this could/should be done).
        """

    def project_page_graphs(self):
        rval = []

        descr = f"""
        Suppose that the fleet size is constant.
        """
        rval.append(dict(
            sts_key='n_pacific_log_tugs',
            t_unit='years',
            descr=descr))

        descr = f"""
        Suppose that the number of ZEV vessels increases linearly at a steady
        rate sufficient to replace the fleet every 20 years.
        """
        rval.append(dict(
            sts_key='n_pacific_log_tugs_ZEV',
            t_unit='years',
            descr=descr))

        descr = """
        The impact on CO2 emissions from the Pacific log tug fleet is expected to be significant.
        """
        rval.append(dict(
            sts_key='pacific_log_barge_CO2',
            t_unit='years',
            figtype='plot vs baseline',
            descr=descr))

        descr = """
        The impact on atmospheric CO2 concentration is expected to be very small.
        That said, if the technology used in this solution were scaled globally,
        the impact could be significant. (TODO: estimate how significant)
        """
        rval.append(dict(
            sts_key='Atmospheric_CO2_conc',
            t_unit='years',
            figtype='plot vs baseline',
            descr=descr))

        descr = """
        A visualization of the difference in global heat forcing reveals the shape of the impact over time.
        The datapoints in this curve are used to compute the Net Present Heat for the project, by adding up the energy associated with each year (modulated by the future discount factor).
        """
        rval.append(dict(
            sts_key='DeltaF_forcing',
            t_unit='years',
            figtype='plot delta',
            descr=descr))

        descr = """
        In other terms, the difference in global heat forcing can be quantified as a small change in (upward) temperature trajectory for the top 200m of the world's oceans.
        """
        rval.append(dict(
            sts_key='Ocean_Temperature_Anomaly',
            t_unit='years',
            figtype='plot delta',
            descr=descr))

        descr = f"""
        In terms of financial modelling, the project is modelled as
        <ul>
        <li>needing {self.r_and_d_annual_cost.to(u.megaCAD/u.year)} for research, development, and operations,</li>
        <li>needing {self.cost_per_tug.to(u.megaCAD)} per vessel at a rate sufficient to replace the fleet over vessel lifetime of {self.vessel_lifetime}
        <li>earning {(self.fuel_cost_rate * u.year).to(u.megaCAD)} annually per vessel [set] in saved fuel</li>
        </ul>
        """
        rval.append(dict(
            sts_key=self.after_tax_cashflow_name,
            t_unit='years',
            figtype='plot',
            descr=descr))
        return rval
