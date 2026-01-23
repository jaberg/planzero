# TODO:
# For more accurate climate simulation, check out
# https://climate-assessment.readthedocs.io/en/latest/index.html

import array
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
from pydantic import BaseModel

ureg = pint.UnitRegistry()
ureg.define('CAD = [currency]')
ureg.define('USD = 1.35 CAD')

ureg.define('people = [human_population]')
ureg.define('cattle = [bovine_population]')

ureg.define('fraction = [] = frac')
ureg.define('ppm = 1e-6 fraction')
ureg.define('ppb = 1e-9 fraction')
u = ureg

_seconds_per_year = (1 * u.year).to(u.second).magnitude

# TODO: a global table of official floating-point values of year-start times,
#       for use by annual step functions, accounting math etc.
#       to avoid floating point rounding errors where years are supposed to line up
#       Same for months, maybe weeks.


from . import ipcc_canada
GHGs = ('CO2', 'CH4', 'N2O', 'HFC', 'PFC', 'SF6', 'NF3')

class SparseTimeSeries(BaseModel):
    """A data structure of (time, value) pairs (stored separately) representing
    a timeseries. It may or not have a default value.
    """

    t_unit:object
    v_unit:object

    times:object # will be a float array
    values:object # will be a float array

    current_readers:list[str]
    writer:str|None

    identifier: str | None # shouldn't be None after construction

    def max(self, _i_start=None):
        if _i_start is None:
            rval = max(self.values)
        else:
            rval = max(self.values[_i_start:])
        return rval * self.v_unit

    def __string__(self):
        return f'STS(id={self.identifier})'

    def _init_v_unit(self, values, unit, default_value):
        # specify unit if you want no default value and you don't know any values yet
        # provide default_value if you do want a default value to apply prior to the first (time, value) pair, then no unit
        if default_value is None:
            # without a default_unit, it's required to either
            # (a) set the unit, or
            # (b) provide initial values and times
            # and it is okay to do both, but then `unit` takes precedence.
            if unit is None:
                return values[0].u
            else:
                if isinstance(unit, str):
                    unit = getattr(u, unit)
                return unit
        else:
            return default_value.u


    def __init__(self, *, times=None, values=None, unit=None, identifier=None, t_unit=u.seconds, default_value=None):
        super().__init__(
            t_unit=t_unit,
            v_unit=self._init_v_unit(values, unit, default_value),
            times=array.array('d'),
            values=array.array('d'),
            current_readers=[],
            writer=None,
            identifier=identifier)
        if default_value is None:
            self.values.append(float('nan'))
        else:
            self.values.append(default_value.to(self.v_unit).magnitude)
        if times is not None:
            self.extend(times, values)

    def __len__(self):
        return len(self.times)

    def _idx_of_time(self, t_query, inclusive):
        """Return the index into the `values` array corresponding to time t_query.
        inclusive=False means the most recent value up to but including t_query

        N.B. that this function can return different values after appending or
        extending the timeseries.
        """
        ts = t_query.to(self.t_unit).magnitude
        if self.times:
            if ts > self.times[-1]:
                index = len(self.values) - 1
            elif ts == self.times[-1] and inclusive:
                index = len(self.values) - 1
            elif ts == self.times[-1] and not inclusive:
                index = len(self.values) - 2
            else:
                if inclusive:
                    index = bisect.bisect_right(self.times, ts)
                else:
                    index = bisect.bisect_left(self.times, ts)
        else:
            index = 0
        return index

    def query(self, t_query, inclusive):
        try:
            n_queries = len(t_query)
        except:
            n_queries = 1
        if n_queries > 1:
            numbers = [self.values[self._idx_of_time(tqi, inclusive)]
                       for tqi in t_query]
            return np.asarray(numbers) * self.v_unit
        else:
            return self.values[self._idx_of_time(t_query, inclusive)] * self.v_unit

    def append(self, t, v):
        if t.u == self.t_unit:
            tt = t.magnitude
        elif t.u == u.year and self.t_unit == u.second:
            tt = t.magnitude * _seconds_per_year
        else:
            tt = t.to(self.t_unit).magnitude
        if len(self.times):
            assert tt > self.times[-1]
        self.times.append(tt)
        self.values.append(v.to(self.v_unit).magnitude)

    def extend(self, times, values):
        assert len(times) == len(values)
        for t, v in zip(times, values):
            self.append(t, v)

    def plot(self, t_unit=None, annotate=True, **kwargs):
        t_unit = t_unit or self.t_unit
        plt.scatter(
            self.times,
            self.values[1:],
            **kwargs)
        plt.xlabel(t_unit)
        plt.ylabel(self.v_unit)
        plt.title(self.identifier)
        if annotate:
            self.annotate_plot(t_unit=t_unit, **kwargs)

    def annotate_plot(self, t_unit=None, **kwargs):
        """Called once per variable name in comparison plots"""
        pass


class Project(BaseModel):

    identifier: str
    _sub_projects: 'list[Project]' = []

    may_register_emissions:bool = True
    requires_emissions_registration_closed:bool = False

    def __init__(self, **kwargs):
        if 'identifier' not in kwargs:
            kwargs = dict(kwargs, identifier=self.__class__.__name__)
        super().__init__(**kwargs)

    def init_add_subprojects(self, sub_projects):
        self._sub_projects.extend(sub_projects)

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
            sts.plot(t_unit=config.get('t_unit'), label=self.title)
        elif config.get('figtype') == 'plot vs baseline':
            sts = state.sts[key]
            sts.plot(t_unit=config.get('t_unit'), label=self.title)
            comparison.state_B.sts[key].plot(
                t_unit=config.get('t_unit'),
                label='Baseline',
                )
            plt.legend(loc='lower left')
        elif config.get('figtype') == 'plot delta':
            years = comparison._years()
            vals_A = comparison.state_A.sts[key].query(years, inclusive=True)
            vals_B = comparison.state_B.sts[key].query(years, inclusive=True)
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


BaseScenario_subclasses = []

class BaseScenarioProject(Project):
    """Inherit from BaseScenarioProject to be included in the default base
    scenario for project evaluation.
    """

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        BaseScenario_subclasses.append(cls)

    @staticmethod
    def base_scenario_projects():
        return [cls() for cls in BaseScenario_subclasses]


class StateCurrent(object):
    def __init__(self, state, readable, writeable):
        self.__dict__.update(
            state=state,
            readable=readable,
            writeable=writeable)

    def __getattr__(self, attr):
        if attr in self.readable or attr in self.writeable:
            # TODO: it might catch errors to be strict about not reading
            # before writing but current.foo += 1 is such natural syntax
            # and strict semantics would forbid it.
            return self.state.sts[attr].query(self.state.t_now, inclusive=True)
        else:
            if attr in self.state.sts:
                raise AttributeError(f'state variable {attr} exists, but the calling Project class did not register to read it')
            else:
                raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if attr in self.writeable:
            self.state.sts[attr].append(self.state.t_now, val)
            self.readable.add(attr)
            self.writeable.remove(attr)
        else:
            assert 0, ('Setting non-writeable attr', attr)


class State(object):
    t_start = 1990 * u.years

    def __init__(self, t_start=t_start, name=None):
        self.t_start = t_start
        self._t_now = t_start
        self.sts = {}
        self.sectoral_emissions_contributors = {
            catpath: {} for catpath in sorted(ipcc_canada.catpaths)}
        self.projects = {}
        self.project_writes = {} # prj.identifier -> set of string names
        self.project_requires_current = {} # prj.identifier -> set of string names
        self.project_t_next = {} # prj.identifier -> t_next
        self._depgraph = None
        self.name = name
        self.emissions_registration_closed = False

    def dependency_digraph(self):
        graph = nx.DiGraph()
        things = set()
        things.update(sts.identifier for sts in self.sts.values())
        things.update(prj.identifier for prj in self.projects.values())
        assert len(things) == len(self.sts) + len(self.projects)
        for sts in self.sts.values():
            graph.add_node(sts.identifier)
            if sts.writer:
                graph.add_edge(sts.writer.identifier, sts.identifier)
            for prj in sts.current_readers:
                graph.add_edge(sts.identifier, prj.identifier)
        for prj in self.projects.values():
            graph.add_node(prj.identifier)
        return graph

    @contextlib.contextmanager
    def requiring_latest(self, project):
        class Context(object):
            def __setattr__(_, name, sts):
                if name not in self.sts:
                    self.sts[name] = sts
                    self._depgraph = None
                    sts.identifier = name
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
                self.sts[name].current_readers.append(project)
                self.project_requires_current[project.identifier].add(name)
                self._depgraph = None

            def __setattr__(_, name, sts):
                if name not in self.sts:
                    self.sts[name] = sts
                    sts.identifier = name
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
                    sts.identifier = name
                assert self.sts[name].writer is None
                self.sts[name].writer = project
                self.project_writes[project.identifier].add(name)
                self._depgraph = None
                return self.sts[name]
        try:
            yield Context()
        finally:
            pass

    def add_project(self, project):
        assert project.identifier not in self.projects
        self.projects[project.identifier] = project
        self.project_writes[project.identifier] = set() # of strings
        self.project_requires_current[project.identifier] = set() # of strings
        self.project_t_next[project.identifier] = project.on_add_project(self)
        self._depgraph = None

        # we close registration at the request of the first project that
        # requires it to be closed
        self.emissions_registration_closed |= project.requires_emissions_registration_closed

        self.add_projects(project._sub_projects)

    def add_projects(self, projects):
        # start with projects that may register emissions
        # so we can schedule the ones that may require emissions registration
        # to be closed at the end (i.e. AtmosphericChemistry)
        order = [(0 if proj.may_register_emissions else 1, ii, proj)
                     for ii, proj in enumerate(projects)]
        order.sort()
        for _, _, project in order:
            self.add_project(project)

    @property
    def t_now(self):
        return self._t_now

    @t_now.setter
    def t_now(self, t_next):
        assert t_next >= self._t_now
        self._t_now = t_next

    def register_emission(self, category_path, ghg, sts_key):
        if self.emissions_registration_closed:
            raise RuntimeError()
        assert ghg in GHGs
        assert sts_key in self.sts
        self.sectoral_emissions_contributors[category_path].setdefault(ghg, []).append(sts_key)

    @property
    def latest(self):
        class Latest(object):
            def __getattr__(_, attr):
                return self.sts[attr].query(self.t_now, inclusive=False)
        return Latest()

    def _current(self, readable_attrs, writeable_attrs):
        """Return a view of certain state variables, supporting the standard
        syntax of step(state, current).

        The semantics of this object are such that
        `current.foo` returns the self.t_now'th value of foo, when it is valid to do so.
        """
        readable = set(readable_attrs)
        writeable = set(writeable_attrs)
        return StateCurrent(self, readable, writeable)

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
                (self.project_t_next[prj_identifier], ii, prj_identifier)
                for (ii, prj_identifier) in enumerate(nx.topological_sort(self._depgraph))
                if self.project_t_next.get(prj_identifier) is not None]
            heapq.heapify(self._heap)

        while self.t_now <= t_stop and self._heap:
            t_next, node_idx, prj_identifier = heapq.heappop(self._heap)
            assert t_next >= self.t_now
            self.t_now = t_next
            if 0:
                print("HEAP")
                for foo in sorted(self._heap):
                    print("    ", foo)
                print('Stepping', t_next, prj_identifier)

            current = self._current(
                    readable_attrs=self.project_requires_current[prj_identifier],
                    writeable_attrs=self.project_writes[prj_identifier])
            new_t_next = self.projects[prj_identifier].step(self, current=current)
            self.project_t_next[prj_identifier] = new_t_next
            if new_t_next is not None:
                assert new_t_next > self.t_now
                heapq.heappush(self._heap, (new_t_next, node_idx, prj_identifier))


class GlobalHeatEnergy(SparseTimeSeries):
    pass

    #def annotate_plot(self, t_unit=None, **kwargs):
        #height = specific_heat_of_top_300m_of_ocean.to(self.v_unit / u.kelvin).magnitude
        #plt.axhline(height)
        #plt.text(self.times[0].to('years').magnitude, height, "Shallow Ocean 1C")


surface_area_of_earth = 5.1e14 * u.m * u.m

molar_mass_CH4 = 16.0 * u.g / u.mol
molar_mass_CO2 = 44.0 * u.g / u.mol
molar_mass_N2O = 44.01 * u.g / u.mol

atmospheric_conc_per_mass_CO2 = (1 * u.ppm) / (7.8 * u.gigatonne)
atmospheric_conc_per_mass_CH4 = atmospheric_conc_per_mass_CO2 * molar_mass_CO2 / molar_mass_CH4
atmospheric_conc_per_mass_N2O = atmospheric_conc_per_mass_CO2 * molar_mass_CO2 / molar_mass_N2O
atmospheric_conc_per_mass_HFC = (1 * u.ppb) / (18.0 * u.megatonne) # HFC-134a (most common HFC)
atmospheric_conc_per_mass_PFC = (1 * u.ppb) / (15.6 * u.megatonne) # CF4 (most common PFC)
atmospheric_conc_per_mass_SF6 = (1 * u.ppb) / (25.9 * u.megatonne)
atmospheric_conc_per_mass_NF3 = (1 * u.ppb) / (12.6 * u.megatonne)


deltaF_coef_N2O = 0.12 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_HFC = 0.16 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_PFC = 0.08 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_SF6 = 0.57 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_NF3 = 0.21 * u.watt / (u.m * u.m) * surface_area_of_earth

CH4_GWP_100 = 28.0 # global warming potential, for CO2e calculation
N2O_GWP_100 = 265.0 # global warming potential, for CO2e calculation
HFC_GWP_100 = 1_430
PFC_GWP_100 = 6_630
SF6_GWP_100 = 23_500
NF3_GWP_100 = 17_200



class AtmosphericChemistry(BaseScenarioProject):
    methane_decay_timescale:float = 10.0

    may_register_emissions:bool = False
    requires_emissions_registration_closed:bool = True

    stepsize:object
    decay_N2O:object
    decay_HFC:object
    decay_PFC:object
    decay_SF6:object
    decay_NF3:object

    def __init__(self, stepsize=1.0 * u.years):
        super().__init__(
            stepsize=stepsize,
            decay_N2O=(1 - stepsize / (114 * u.years)),
            decay_HFC=(1 - stepsize / (14 * u.years)),
            decay_PFC=1.0,
            decay_SF6=1.0,
            decay_NF3=1.0,
            )

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            for catpath, contributors in state.sectoral_emissions_contributors.items():
                for sts_key in contributors.get('CO2', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('CH4', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('N2O', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('HFC', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('PFC', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('SF6', []):
                    ctx.will_read_current(sts_key)
                for sts_key in contributors.get('NF3', []):
                    ctx.will_read_current(sts_key)

        with state.defining(self) as ctx:
            for catpath, _ in state.sectoral_emissions_contributors.items():
                setattr(ctx, f'Predicted_Annual_Emitted_CO2_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_CH4_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_N2O_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_HFC_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_PFC_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_SF6_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_NF3_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))
                setattr(ctx, f'Predicted_Annual_Emitted_CO2e_mass_{catpath}', SparseTimeSeries(unit=u.kiloton))

            ctx.Predicted_Annual_Emitted_CO2_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_CH4_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_N2O_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_HFC_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_PFC_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_SF6_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_NF3_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Predicted_Annual_Emitted_CO2e_mass = SparseTimeSeries(unit=u.kiloton)

            # XXX : what year do these numbers represent? How can this be a default value
            # when the simulated years are a parameter of the state?
            ctx.Atmospheric_CO2_conc = SparseTimeSeries(unit=u.ppm, default_value=400.0 * u.ppm)
            ctx.Atmospheric_CH4_conc = SparseTimeSeries(unit=u.ppb, default_value=1775.0 * u.ppb)
            ctx.Atmospheric_N2O_conc = SparseTimeSeries(unit=u.ppb, default_value=336.0 * u.ppb)

            # Gemini says these data are from NOAA and are accurate for January 2026
            ctx.Atmospheric_HFC_conc = SparseTimeSeries(unit=u.ppb, default_value=0.1345 * u.ppb)
            ctx.Atmospheric_PFC_conc = SparseTimeSeries(unit=u.ppb, default_value=0.0902 * u.ppb)
            ctx.Atmospheric_SF6_conc = SparseTimeSeries(unit=u.ppb, default_value=0.0124 * u.ppb)
            ctx.Atmospheric_NF3_conc = SparseTimeSeries(unit=u.ppb, default_value=0.0036 * u.ppb)

            ctx.DeltaF_CO2 = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_CH4 = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_N2O = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_HFC = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_PFC = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_SF6 = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_NF3 = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_forcing = SparseTimeSeries(unit=u.petawatt)
            ctx.DeltaF_feedback = SparseTimeSeries(unit=u.petawatt)

            # Heat Energy forcing is the heat equivalent to net annual cashflow, an annual integral
            ctx.Annual_Heat_Energy_forcing = SparseTimeSeries(default_value=0 * u.exajoule)
            ctx.Cumulative_Heat_Energy_forcing = SparseTimeSeries(default_value=0 * u.exajoule)
            ctx.Heat_Energy_imbalance = SparseTimeSeries(unit=u.exajoule)
            ctx.Cumulative_Heat_Energy = GlobalHeatEnergy(default_value=0.0 * u.exajoule)
            ctx.Ocean_Temperature_Anomaly = SparseTimeSeries(default_value=1.3 * u.kelvin)

        return state.t_now

    def step(self, state, current):
        # add up annual emissions from registry
        annual_CO2_mass = 0 * u.kiloton
        annual_CH4_mass = 0 * u.kiloton
        annual_N2O_mass = 0 * u.kiloton
        annual_HFC_mass = 0 * u.kiloton
        annual_PFC_mass = 0 * u.kiloton
        annual_SF6_mass = 0 * u.kiloton
        annual_NF3_mass = 0 * u.kiloton

        for catpath, contributors in state.sectoral_emissions_contributors.items():
            catpath_CO2e_mass = 0 * u.kg

            catpath_CO2_contributors = contributors.get('CO2', [])
            if catpath_CO2_contributors:
                catpath_CO2_mass = sum(getattr(current, sts_key) for sts_key in catpath_CO2_contributors)
                setattr(current, f'Predicted_Annual_Emitted_CO2_mass_{catpath}', catpath_CO2_mass)
                catpath_CO2e_mass += catpath_CO2_mass
                annual_CO2_mass += catpath_CO2_mass

            catpath_CH4_contributors = contributors.get('CH4', [])
            if catpath_CH4_contributors:
                catpath_CH4_mass = sum(getattr(current, sts_key) for sts_key in catpath_CH4_contributors)
                setattr(current, f'Predicted_Annual_Emitted_CH4_mass_{catpath}', catpath_CH4_mass)
                catpath_CO2e_mass += catpath_CH4_mass * CH4_GWP_100
                annual_CH4_mass += catpath_CH4_mass

            catpath_N2O_contributors = contributors.get('N2O', [])
            if catpath_N2O_contributors:
                catpath_N2O_mass = sum(getattr(current, sts_key) for sts_key in catpath_N2O_contributors)
                setattr(current, f'Predicted_Annual_Emitted_N2O_mass_{catpath}', catpath_N2O_mass)
                catpath_CO2e_mass += catpath_N2O_mass * N2O_GWP_100
                annual_N2O_mass += catpath_N2O_mass

            catpath_HFC_contributors = contributors.get('HFC', [])
            if catpath_HFC_contributors:
                catpath_HFC_mass = sum(getattr(current, sts_key) for sts_key in catpath_HFC_contributors)
                setattr(current, f'Predicted_Annual_Emitted_HFC_mass_{catpath}', catpath_HFC_mass)
                catpath_CO2e_mass += catpath_HFC_mass * HFC_GWP_100
                annual_HFC_mass += catpath_HFC_mass

            catpath_PFC_contributors = contributors.get('PFC', [])
            if catpath_PFC_contributors:
                catpath_PFC_mass = sum(getattr(current, sts_key) for sts_key in catpath_PFC_contributors)
                setattr(current, f'Predicted_Annual_Emitted_PFC_mass_{catpath}', catpath_PFC_mass)
                catpath_CO2e_mass += catpath_PFC_mass * PFC_GWP_100
                annual_PFC_mass += catpath_PFC_mass

            catpath_SF6_contributors = contributors.get('SF6', [])
            if catpath_SF6_contributors:
                catpath_SF6_mass = sum(getattr(current, sts_key) for sts_key in catpath_SF6_contributors)
                setattr(current, f'Predicted_Annual_Emitted_SF6_mass_{catpath}', catpath_SF6_mass)
                catpath_CO2e_mass += catpath_SF6_mass * SF6_GWP_100
                annual_SF6_mass += catpath_SF6_mass

            catpath_NF3_contributors = contributors.get('NF3', [])
            if catpath_NF3_contributors:
                catpath_NF3_mass = sum(getattr(current, sts_key) for sts_key in catpath_NF3_contributors)
                setattr(current, f'Predicted_Annual_Emitted_NF3_mass_{catpath}', catpath_NF3_mass)
                catpath_CO2e_mass += catpath_NF3_mass * NF3_GWP_100
                annual_NF3_mass += catpath_NF3_mass

            setattr(current, f'Predicted_Annual_Emitted_CO2e_mass_{catpath}', catpath_CO2e_mass)


        current.Predicted_Annual_Emitted_CO2_mass = annual_CO2_mass
        current.Predicted_Annual_Emitted_CH4_mass = annual_CH4_mass
        current.Predicted_Annual_Emitted_N2O_mass = annual_N2O_mass
        current.Predicted_Annual_Emitted_HFC_mass = annual_HFC_mass
        current.Predicted_Annual_Emitted_PFC_mass = annual_PFC_mass
        current.Predicted_Annual_Emitted_SF6_mass = annual_SF6_mass
        current.Predicted_Annual_Emitted_NF3_mass = annual_NF3_mass
        current.Predicted_Annual_Emitted_CO2e_mass = (
            annual_CO2_mass
            + CH4_GWP_100 * annual_CH4_mass
            + N2O_GWP_100 * annual_N2O_mass
            + HFC_GWP_100 * annual_HFC_mass
            + PFC_GWP_100 * annual_PFC_mass
            + SF6_GWP_100 * annual_SF6_mass
            + NF3_GWP_100 * annual_NF3_mass
        )

        fraction_of_emitted_CO2_that_becomes_atmospheric = .45

        # apply an atmospheric climate model
        annual_CO2_mass_atmospheric = (
            annual_CO2_mass
            * fraction_of_emitted_CO2_that_becomes_atmospheric)
        annual_CH4_mass_atmospheric = annual_CH4_mass * 1.0 # no such discounting of CH4

        annual_emitted_CO2_in_atmosphere_as_concentration = (
            annual_CO2_mass_atmospheric
            * atmospheric_conc_per_mass_CO2)

        annual_emitted_CH4_in_atmosphere_as_concentration = (
            annual_CH4_mass_atmospheric
            / (2.78 * u.megatonne / u.ppb))

        # TODO this should be multiplied by stepsize, not 1 year implicitly,
        #      and this process should be tested for robustness to step size
        tau_ch4 = 12.0 # years
        annual_ch4_to_co2_decay = (
            state.latest.Atmospheric_CH4_conc
            / tau_ch4)

        current.Atmospheric_CH4_conc += (
            annual_emitted_CH4_in_atmosphere_as_concentration
            + 180 * u.ppb # baseline from other sources
            - annual_ch4_to_co2_decay)

        # no decay is assumed for CO2
        current.Atmospheric_CO2_conc += (
            annual_emitted_CO2_in_atmosphere_as_concentration
            + 2 * u.ppm # baseline from other sources
            + (annual_ch4_to_co2_decay
               * fraction_of_emitted_CO2_that_becomes_atmospheric)
        )

        reference_CO2_conc = 280.0 * u.ppm
        current.DeltaF_CO2 = (
            5.35 * u.watt / (u.m * u.m)
            * surface_area_of_earth
            * np.log(current.Atmospheric_CO2_conc.to(u.ppm).magnitude
                     / reference_CO2_conc.to(u.ppm).magnitude))

        reference_CH4_conc = 722.0 * u.ppb
        current.DeltaF_CH4 = (
            0.036 * u.watt / (u.m * u.m)
            * surface_area_of_earth
            * (np.sqrt(current.Atmospheric_CH4_conc.to(u.ppb).magnitude)
               - np.sqrt(reference_CH4_conc.to(u.ppb).magnitude)))

        self.step_N2O(state, current, annual_N2O_mass)
        self.step_HFC(state, current, annual_HFC_mass)
        self.step_PFC(state, current, annual_PFC_mass)
        self.step_SF6(state, current, annual_SF6_mass)
        self.step_NF3(state, current, annual_NF3_mass)

        current.DeltaF_forcing = (
            current.DeltaF_CO2
            + current.DeltaF_CH4
            + current.DeltaF_N2O
            + current.DeltaF_HFC
            + current.DeltaF_PFC
            + current.DeltaF_SF6
            + current.DeltaF_NF3
        )

        current.DeltaF_feedback = (
            -1.3 * u.watt / (u.m * u.m) / u.kelvin
            * surface_area_of_earth
            * current.Ocean_Temperature_Anomaly) # will be stale value

        current.Annual_Heat_Energy_forcing = (
            self.stepsize # integrate over duration of stepsize aka 1 year
            * current.DeltaF_forcing)

        current.Cumulative_Heat_Energy_forcing += (
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

    def step_N2O(self, state, current, annual_N2O_mass):
        conc = current.Atmospheric_N2O_conc

        conc += atmospheric_conc_per_mass_N2O * annual_N2O_mass
        conc *= self.decay_N2O
        current.Atmospheric_N2O_conc = conc

        reference_N2O_conc = 270.0 * u.ppb

        current.DeltaF_N2O = (
            deltaF_coef_N2O
            * (np.sqrt(conc.to(u.ppb).magnitude)
               - np.sqrt(reference_N2O_conc.to(u.ppb).magnitude)))

    def step_HFC(self, state, current, annual_HFC_mass):
        conc = current.Atmospheric_HFC_conc

        conc += atmospheric_conc_per_mass_HFC * annual_HFC_mass
        conc *= self.decay_HFC
        current.Atmospheric_HFC_conc = conc

        current.DeltaF_HFC = deltaF_coef_HFC * conc.to(u.ppb).magnitude

    def step_PFC(self, state, current, annual_PFC_mass):
        conc = current.Atmospheric_PFC_conc

        conc += atmospheric_conc_per_mass_PFC * annual_PFC_mass
        conc *= self.decay_PFC
        current.Atmospheric_PFC_conc = conc

        current.DeltaF_PFC = deltaF_coef_PFC * conc.to(u.ppb).magnitude

    def step_SF6(self, state, current, annual_SF6_mass):
        conc = current.Atmospheric_SF6_conc

        conc += atmospheric_conc_per_mass_SF6 * annual_SF6_mass
        conc *= self.decay_SF6
        current.Atmospheric_SF6_conc = conc

        current.DeltaF_SF6 = deltaF_coef_SF6 * conc.to(u.ppb).magnitude

    def step_NF3(self, state, current, annual_NF3_mass):
        conc = current.Atmospheric_NF3_conc

        conc += atmospheric_conc_per_mass_NF3 * annual_NF3_mass
        conc *= self.decay_NF3
        current.Atmospheric_NF3_conc = conc

        current.DeltaF_NF3 = deltaF_coef_NF3 * conc.to(u.ppb).magnitude

class GeometricHumanPopulationForecast(BaseScenarioProject):
    rate:float = 1.014
    stepsize:object = 1.0 * u.years

    def on_add_project(self, state):
        assert 1989 * u.years <= state.t_now <= 1991 * u.years, state.t_now.to(u.years)

        with state.defining(self) as ctx:
            ctx.human_population = SparseTimeSeries(
                times=[state.t_now],
                values=[27_685_730 * u.people])

        return state.t_now + self.stepsize

    def step(self, state, current):
        current.human_population *= self.rate
        return state.t_now + self.stepsize


class GeometricBovinePopulationForecast(BaseScenarioProject):

    # https://www.ucdavis.edu/food/news/making-cattle-more-sustainable
    # 70% of emissions remain, according to https://www.helsinki.fi/en/news/climate-change/new-feed-additive-can-significantly-reduce-methane-emissions-generated-ruminants-already-dairy-farm
    # https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html
    bovaer_methane_reduction_fraction:float = .625
    # TODO: dairy cattle average is .7
    # TODO: beef cattle reduction is greater, multiplier should be .55

    stepsize:object = 1.0 * u.years
    methane_per_head_per_year:object = 175 * u.pounds / u.cattle

    jan1:object = None
    jul1:object = None

    def __init__(self):
        super().__init__()
        csv = pd.read_csv(os.path.join(os.environ['PLANZERO_DATA'], 'number-of-cattle.csv'))
        self.jan1 = csv[(csv['Farm type'] == 'On all cattle operations') & (csv['Survey date'] == 'At January 1')]
        self.jul1 = csv[(csv['Farm type'] == 'On all cattle operations') & (csv['Survey date'] == 'At July 1')]

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = SparseTimeSeries(
                default_value=0 * u.dimensionless)
        with state.defining(self) as ctx:
            ctx.bovine_population_on_bovaer = SparseTimeSeries(
                default_value=0 * u.cattle)
            ctx.bovine_population = SparseTimeSeries(
                times=self.jan1['REF_DATE'].values * u.years,
                values=(.5 * self.jan1['VALUE'].values * 1000
                         + .5 * self.jul1['VALUE'].values * 1000) * u.cattle)
            ctx.bovine_methane = SparseTimeSeries(
                default_value=(
                    state.sts['bovine_population'].query(2000 * u.years, inclusive=True)
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
        years = np.arange(start_year, stop_year) * u.years
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
        return np.asarray(envelope)

    def net_present_discounted_sum(self, base_rate, key, inclusive=True):
        years = self._years()
        vals_A = self.state_A.sts[key].query(years, inclusive=inclusive)
        vals_B = self.state_B.sts[key].query(years, inclusive=inclusive)
        diff = vals_A - vals_B
        envelope = self._net_present_envelope(years, base_rate)
        return np.cumsum(diff.magnitude * envelope)[-1] * diff.u

    def net_present_CO2e(self, base_rate):
        return self.net_present_discounted_sum(
            base_rate,
            key='Predicted_Annual_Emitted_CO2e_mass')

    def net_present_heat(self, base_rate):
        return self.net_present_discounted_sum(
            base_rate,
            key='Annual_Heat_Energy_forcing')

    def net_present_value(self, base_rate):
        if self.project.after_tax_cashflow_name in self.state_B.sts:
            raise NotImplementedError()
        years = self._years()
        envelope = self._net_present_envelope(years, base_rate)
        cashflow = self.state_A.sts[self.project.after_tax_cashflow_name].query(
            years, inclusive=True)
        return np.cumsum(cashflow.magnitude * envelope)[-1] * cashflow.u

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
        predictions = state.sts[f'Predicted_Annual_Emitted_CO2e_mass_{catpath}'].query(
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


class IPCC_Forest_Land_Model(BaseScenarioProject):
    stepsize:object = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:

            ctx.Other_Forest_Land_CO2 = SparseTimeSeries(
                default_value=40.0 * u.Mt)

        state.register_emission('Forest_Land', 'CO2', 'Other_Forest_Land_CO2')


class IPCC_Transport_RoadTransportation_LightDutyGasolineTrucks(BaseScenarioProject):
    stepsize:object = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.human_population = SparseTimeSeries(
                times=[state.t_now],
                values=[27_685_730 * u.people])
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
