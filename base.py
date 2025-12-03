import bisect
import contextlib
import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pint
import torch
import time
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

    @property
    def name(self):
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, name):
        self._name = name

    def __string__(self):
        return f'Project(name={self.name})'

    def on_add_project(self, state):
        return state.t_now # return the first time to run

    def step(self, state):
        return None # return t_next to be called again, None to be left alone


class State(object):
    t_start = 2000 * u.years

    def __init__(self, t_start=t_start, name=None):
        self.t_start = t_start
        self._t_now = torch.tensor(t_start.to(u.seconds).magnitude) * u.seconds
        self.sts = {}
        self.sectoral_emissions_contributors = {
            'Enteric Fermentation': {},
        }
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
    methane_GWP = 28.0
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
            ctx.Annual_Emitted_CO2_mass = SparseTimeSeries(unit=u.kiloton)
            ctx.Annual_Emitted_CH4_mass = SparseTimeSeries(unit=u.kiloton)

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
            for sts_key in contributors.get('CO2', []):
                annual_CO2_mass += getattr(current, sts_key)

            for sts_key in contributors.get('CH4', []):
                annual_CH4_mass += getattr(current, sts_key)

        current.Annual_Emitted_CO2_mass = annual_CO2_mass
        current.Annual_Emitted_CH4_mass = annual_CH4_mass

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

        specific_heat_of_top_100m_of_ocean = 151200.0 * u.exajoule / u.kelvin
        current.Ocean_Temperature_Anomaly += (
            current.Heat_Energy_imbalance
            / (specific_heat_of_top_100m_of_ocean * 2))

        current.Cumulative_Heat_Energy += current.Heat_Energy_imbalance

        return state.t_now + self.stepsize


class GeometricHumanPopulationForecast(Project):
    def __init__(self, rate=1.014):
        super().__init__()
        self.rate = rate
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        assert state.t_now == 2000 * u.years

        with state.defining(self) as ctx:
            ctx.human_population = SparseTimeSeries(state.t_now, 30_685_730 * u.people)

        return state.t_now + self.stepsize

    def step(self, state, current):
        current.human_population *= self.rate
        return state.t_now + self.stepsize


class GeometricBovinePopulationForecast(Project):

    # 70% of emissions remain, according to https://www.helsinki.fi/en/news/climate-change/new-feed-additive-can-significantly-reduce-methane-emissions-generated-ruminants-already-dairy-farm
    bovaer_methane_reduction_fraction = .7

    methane_per_head_per_year = 220 * u.pounds / u.cattle
    # https://www.ucdavis.edu/food/news/making-cattle-more-sustainable

    def __init__(self):
        super().__init__()
        csv = pd.read_csv('number-of-cattle.csv')
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
        state.register_emission('Enteric Fermentation', 'CH4', 'bovine_methane')
        return 2000 * u.years + self.stepsize

    def step(self, state, current):
        if state.t_now >= 2026 * u.years:
            current.bovine_population = (
                max(
                    13_000_000 * .992 ** (state.t_now.to('years').magnitude - 2010),
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

    def __init__(self, peak_year=2035 * u.year, shoulder_years=5 * u.years):
        super().__init__()
        self.peak_year = peak_year
        self.shoulder_years = shoulder_years
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        self.t_next = state.t_now
        with state.requiring_latest(self) as ctx:
            ctx.bovine_population_on_bovaer = SparseTimeSeries(
                default_value=0 * u.cattle)
        with state.defining(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.CAD))
        return state.t_now

    def step(self, state, current):
        zval = (
            (state.t_now - self.peak_year)
            / (self.shoulder_years / 2.0))
        current.bovine_population_fraction_on_bovaer = torch.sigmoid(
            zval.to('dimensionless').magnitude) * u.dimensionless
            
        setattr(
            current,
            self.after_tax_cashflow_name, 
            -state.latest.bovine_population_on_bovaer
            * self.bovaer_price * self.stepsize)

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
        years = torch.arange(start_year, stop_year) * u.years
        return years

    def _net_present_envelope(self, years, base_rate):
        present_year_int = int(self.present.to('years').magnitude)
        envelope = [0] * len(years)
        for ii, year in enumerate(years):
            year_int = int(year.to('years').magnitude)
            if year_int >= present_year_int:
                envelope[ii] = base_rate ** (year_int - present_year_int)
        return torch.tensor(envelope)

    def net_present_heat(self, base_rate):
        key = 'Heat_Energy_forcing'
        years = self._years()
        vals_A = self.state_A.sts[key].latest_vals(years, inclusive=True)
        vals_B = self.state_B.sts[key].latest_vals(years, inclusive=True)
        heat_forcing = vals_A - vals_B
        envelope = self._net_present_envelope(years, base_rate)
        return torch.cumsum(heat_forcing.magnitude * envelope, dim=0)[-1] * heat_forcing.u

    def net_present_value(self, base_rate):
        if self.project.after_tax_cashflow_name in self.state_B.sts:
            raise NotImplementedError()
        years = self._years()
        envelope = self._net_present_envelope(years, base_rate)
        cashflow = self.state_A.sts[self.project.after_tax_cashflow_name].latest_vals(
            years, inclusive=True)
        return torch.cumsum(cashflow.magnitude * envelope, dim=0)[-1] * cashflow.u


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
            nph1 = self.comparisons[eval_name].net_present_heat(base_rate=base_rate)
            npv1 = self.comparisons[eval_name].net_present_value(base_rate=base_rate)
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

