# TODO:
# For more accurate climate simulation, check out
# https://climate-assessment.readthedocs.io/en/latest/index.html

import array
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
from pydantic import BaseModel, computed_field
from .ureg import ureg, kt_by_ghg
u = ureg


# TODO: a global table of official floating-point values of year-start times,
#       for use by annual step functions, accounting math etc.
#       to avoid floating point rounding errors where years are supposed to line up
#       Same for months, maybe weeks.


from . import ipcc_canada
from .enums import (
    GHG, IPCC_Sector, PT, SubsidyPrograms,
    IPCC_Sector_from_catpath_with_whitespace)

from .sts import SparseTimeSeries, STS, InterpolationMode


class DynamicElement(BaseModel):

    identifier: str | None = None
    _sub_projects: 'list[DynamicElement]' = []

    may_register_emissions:bool = True
    requires_emissions_registration_closed:bool = False

    tags: set = set() # eg. barrier, strategy

    @computed_field
    def short_description(self) -> str | None:
        # The intent is for subclasses to over-ride this method.
        return self.__class__.__doc__

    @computed_field
    def description(self) -> str | None:
        # The intent is for subclasses to use this implementation
        # by populating the docstring, but they are welcome to
        # override the method as well.
        doc = self.__class__.__doc__
        if doc:
            rval = doc
        else:
            rval = self.short_description # which may be None
        if rval is None:
            raise RuntimeError()
        else:
            return rval

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.identifier is None:
            self.identifier = self.__class__.__name__

        try:
            # don't use this anymore, call it short_descr
            self.short_descr
            raise Exception()
        except AttributeError:
            pass

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
            vals_A = comparison.state_A.sts[key].query(years)
            vals_B = comparison.state_B.sts[key].query(years)
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

class BaseScenarioProject(DynamicElement):
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

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.tags.add('built-in')


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
            return self.state.sts[attr].query(self.state.t_now)
        else:
            if attr in self.state.sts:
                raise AttributeError(f'state variable {attr} exists, but the calling DynamicElement class did not register to read it')
            else:
                raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if attr in self.writeable:
            self.state.sts[attr].append(self.state.t_now, val)
            self.readable.add(attr)
            self.writeable.remove(attr)
        else:
            assert 0, ('Setting non-writeable attr', attr)


class DeclarationContext(object):

    def __init__(self, state, dynelem, need_current, write):
        self.__dict__.update(
            state=state,
            dynelem=dynelem,
            need_current=need_current,
            write=write,
            )

    def __setattr__(self, name, thing):
        if isinstance(thing, STS):
            return self.state.declare_sts(
                self.dynelem,
                thing,
                name=name,
                need_current=self.need_current,
                write=self.write)
        else:
            raise TypeError(thing)


class Stash(object):
    pass


class EmissionResults(BaseModel):

    by_sector_ghg_pt_driver: dict[tuple[object, object, object, object], object]

    @property
    def ipcc_sectors(self) -> set[object]:
        return {
            ipcc_sector
            for (ipcc_sector, _, _, _) in self.by_sector_ghg_pt_driver}

    @property
    def ghgs(self) -> set[object]:
        return {
            ghg
            for (_, ghg, _, _) in self.by_sector_ghg_pt_driver}

    @property
    def pts(self) -> set[object]:
        return {
            pt
            for (_, _, pt, _) in self.by_sector_ghg_pt_driver}

    @property
    def drivers(self) -> set[object]:
        return {
            driver
            for (_, _, _, driver) in self.by_sector_ghg_pt_driver}

    def total(self,
              only_ipcc_sector=None,
              only_driver=None,
              return_None_instead_of_zero=False):
        rval = None
        for ((ipcc_sector, ghg, _, driver), ts) in self.by_sector_ghg_pt_driver.items():
            if only_ipcc_sector is not None and ipcc_sector != only_ipcc_sector:
                continue
            if only_driver is not None and driver != only_driver:
                continue
            co2e = ghgvalues.GWP_100[ghg] * ts
            if rval is None:
                rval = co2e
            else:
                rval += co2e
        if rval is None and not return_None_instead_of_zero:
            return SparseTimeSeries(
                default_value=0 * u.kilotonne_CO2e,
                t_unit=u.year)
        return rval

    def sum(self):
        rval = None
        for ((_, ghg, _, _), ts) in self.by_sector_ghg_pt_driver.items():
            co2e = ghgvalues.GWP_100[ghg] * ts
            if rval is None:
                rval = co2e.sum()
            else:
                rval += co2e.sum()
        if rval is None:
            return 0 * u.kilotonne_CO2e
        return rval


class SubsidyResults(BaseModel):

    by_program_reason_pt_driver: dict[tuple[object, object, object, object], object]

    def total(self):
        rval = None
        for ts in self.by_program_reason_pt_driver.values():
            if rval is None:
                rval = ts
            else:
                rval += ts
        if rval is None:
            return SparseTimeSeries(
                default_value=0 * u.mega_CAD,
                t_unit=u.year)
        return rval

    def sum(self,
            return_None_instead_of_zero=False):
        rval = None
        for ts in self.by_program_reason_pt_driver.values():
            if rval is None:
                rval = ts.sum()
            else:
                rval += ts.sum()
        if rval is None and not return_None_instead_of_zero:
            raise Exception()
        return rval


class State(object):
    t_start = 1990 * u.years

    def __init__(self, t_start=t_start, name=None):
        self.t_start = t_start
        self._t_now = t_start # always in some unit of time, not always the same unit
        self.sts = {}  # the sts objects built up by rolling simulation forward

        self.projects = {}
        self.project_writes = {} # prj.identifier -> set of string names
        self.project_requires_current = {} # prj.identifier -> set of string names
        #self.project_reads = {} # prj.identifier -> set of string names
        self.project_t_next = {} # prj.identifier -> t_next
        self.stashes = {} # prj.identifier -> private namespace
        self._depgraph = None
        self.name = name
        self.emissions_registration_closed = False
        self.registries = dict(
            emission_factor={},
            driver={},
            subsidy_factor={})
        self._computed_annual_emissions = None
        self._computed_annual_subsidies = None
        self.sts_id_counter = 100

    def new_sts_identifier(self):
        name = self.name or 'State_STS'
        rval = f'{name}_{self.sts_id_counter}'
        self.sts_id_counter += 1
        return rval

    def dependency_digraph(self):
        graph = nx.DiGraph()
        things = set()
        things.update(sts.identifier for sts in self.sts.values())
        things.update(prj.identifier for prj in self.projects.values())
        assert len(things) == len(self.sts) + len(self.projects), "It might be confusing to use the same name for an STS and a DynamicElement"
        for sts in self.sts.values():
            graph.add_node(sts.identifier)
            if sts.writer:
                graph.add_edge(sts.writer.identifier, sts.identifier)
            for prj in sts.current_readers:
                graph.add_edge(sts.identifier, prj.identifier)
        for prj in self.projects.values():
            graph.add_node(prj.identifier)
        return graph

    def declare_read_current_sts(self, project, name):
        self.sts[name].current_readers.append(project)
        self.project_requires_current[project.identifier].add(name)
        self._depgraph = None

    def declare_sts(self, project, sts, name=None, need_current=False, write=False):
        if name is None:
            name = sts.identifier or self.new_sts_identifier()
        elif sts.identifier is not None:
            assert name == sts.identifier
        assert isinstance(sts, STS)
        if name not in self.sts:
            assert sts.identifier in (None, name)
            sts.identifier = name
            self.sts[name] = sts
            self._depgraph = None
        else:
            # TODO type-check for compatible existing sts
            pass
        del sts # after this point, we're annotating whatever's in self.sts[name]
        if need_current:
            assert not write
            self.declare_read_current_sts(project, name)
        if write:
            assert not need_current
            assert self.sts[name].writer is None
            self.sts[name].writer = project
            self.project_writes[project.identifier].add(name)
            self._depgraph = None
            return self.sts[name]
        return self.sts[name]

    @contextlib.contextmanager
    def requiring_latest(self, project):
        """ STS declaration syntax for variables whose "latest" values are
        required by a dynamic element. "Latest" values can be used to break
        dependency cycles that would occur if "Current" values were required.
        """
        yield DeclarationContext(self, project, need_current=False, write=False)

    @contextlib.contextmanager
    def requiring_current(self, project):
        """ STS declaration syntax for variables whose "current" values are
        required by a dynamic element. If any of these STS variables have
        a value defined for the same time as the `project` has requested
        a call to step(), step() will only be called after those input values have been determined,
        and these up-to-date input values will be provided to the step function.
        """
        yield DeclarationContext(self, project, need_current=True, write=False)

    @contextlib.contextmanager
    def defining(self, project, catpath=None, ghg=None):
        yield DeclarationContext(self, project, need_current=False, write=True)

    def stash(self, project):
        return self.stashes[project.identifier]

    def add_project(self, project):
        assert project.identifier not in self.projects
        self.projects[project.identifier] = project
        self.project_writes[project.identifier] = set() # of strings
        self.project_requires_current[project.identifier] = set() # of strings
        self.stashes[project.identifier] = Stash()
        self.project_t_next[project.identifier] = project.on_add_project(self)
        self._depgraph = None

        # we will close registration at the request of the first project that
        # requires it to be closed
        # TODO: remind me why this mechanism was required?
        #       I'm guessing it had something to do with collecting registries
        #       Any registry-collecting dynamic element won't know what its
        #       inputs are until all participating dynamic elements have been
        #       added to the state.
        #       A solution would be to support other types of STS container than ObjectTensor
        #       such as list of STS or dict of STS; that way, a dynamic element
        #       could declare e.g. (1) the whole dict of STS as its input.
        #       and instead of registering an emission, a dynamic element could declare
        #       that it's output STS should be a member of such-and-such dictionary.
        #       Getting this to work without abstraction leaks might be tricky, because
        #       in some sense the dictionary is an input, but from a dependency graph
        #       perspective, it's all the elements that are the dependencies, and the
        #       dictionary itself isn't part of the dependency graph.
        #
        #       ObjectTensor container actually didn't work out that well.
        #       Now I'm more optimistic about tagging STS objects by e.g.
        #       CO2_emissions, CH4_emissions, investment_capital, etc.
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

    def add_dynamic_elements(self, dynamic_elements):
        return self.add_projects(dynamic_elements)

    @property
    def t_now(self):
        return self._t_now

    @t_now.setter
    def t_now(self, t_next):
        assert t_next >= self._t_now
        self._t_now = t_next

    def register_emission_factor(self, pt, driver, sts_key, ipcc_sector, ghg):
        if self.emissions_registration_closed:
            raise RuntimeError()
        ipcc_sector = IPCC_Sector(ipcc_sector)
        ghg = GHG(ghg)
        pt = PT(pt)
        assert sts_key in self.sts
        driver_d = self.registries['emission_factor']\
                .setdefault(ghg, {})\
                .setdefault(pt, {}) \
                .setdefault(ipcc_sector, {})
        assert driver not in driver_d
        driver_d[driver] = sts_key

    def register_driver(self, pt, driver, sts_key):
        ts = self.sts[sts_key]
        if ts.interpolation == InterpolationMode.no_interpolation:
            assert ts.t_unit == u.years
            assert all(tt == int(tt) for tt in ts.times)
        else:
            # an integral will be computed later
            pass

        if self.emissions_registration_closed:
            raise RuntimeError()
        pt = PT(pt)
        assert sts_key in self.sts
        driver_d = self.registries['driver']\
                .setdefault(pt, {})
        assert driver not in driver_d
        driver_d[driver] = sts_key

    def register_subsidy_factor(self, pt, driver, sts_key, program, reason):
        if self.emissions_registration_closed:
            raise RuntimeError()
        assert sts_key in self.sts
        pt = PT(pt)
        program = SubsidyPrograms(program)
        reason_d = self.registries['subsidy_factor']\
                .setdefault(program, {})\
                .setdefault(driver, {})\
                .setdefault(pt, {})
        assert reason not in reason_d
        reason_d[reason] = sts_key

    def annual_bin_boundaries(self):
        year_start_int = int(self.t_start.to(u.years).magnitude)
        year_end_float = self._t_now.to(u.years).magnitude
        year_end_int = int(math.ceil(year_end_float))
        boundaries = np.arange(year_start_int, year_end_int + 1) * u.years
        return boundaries

    def compute_annual_emissions(self):
        if self._computed_annual_emissions is not None:
            return self._computed_annual_emissions
        by_sector_ghg_pt_driver = {}
        boundaries = self.annual_bin_boundaries()

        for pt, driver_d in self.registries['driver'].items():
            for driver, driver_key in driver_d.items():
                driver_ts = self.sts[driver_key]
                for ghg, ef_by_pt in self.registries['emission_factor'].items():
                    if pt not in ef_by_pt:
                        print('Warning: missing ef', pt, driver, ghg, ef_by_pt.keys())
                        continue
                    for sector, ef_by_driver in ef_by_pt[pt].items():
                        if driver not in ef_by_driver:
                            # not all drivers drive emissions, some are for e.g. subsidies
                            continue
                        ef_key = ef_by_driver[driver]
                        ef_ts = self.sts[ef_key]
                        # todo: verify driver_ts is annual totals
                        if driver_ts.interpolation == InterpolationMode.no_interpolation:
                            em = ef_ts * driver_ts * (1 * u.year)
                        else:
                            em = (ef_ts * driver_ts).bin_integrals(bin_boundaries=boundaries)
                        by_sector_ghg_pt_driver[sector, ghg, pt, driver] = em.to(
                            kt_by_ghg[ghg])
        self._computed_annual_emissions = EmissionResults(
            by_sector_ghg_pt_driver=by_sector_ghg_pt_driver)
        return self._computed_annual_emissions

    def compute_annual_subsidies(self):
        if self._computed_annual_subsidies is not None:
            return self._computed_annual_subsidies
        by_program_reason_pt_driver = {}
        boundaries = self.annual_bin_boundaries()
        for pt, driver_d in self.registries['driver'].items():
            for driver, driver_key in driver_d.items():
                driver_ts = self.sts[driver_key]
                for program, sf_by_driver in self.registries['subsidy_factor'].items():
                    if driver not in sf_by_driver:
                        # some drivers are just for emissions
                        continue
                    for reason, sf_key in sf_by_driver[driver][pt].items():
                        sf_ts = self.sts[sf_key]
                        # todo: verify driver_ts is annual totals
                        if driver_ts.interpolation == InterpolationMode.no_interpolation:
                            subsidies = sf_ts * driver_ts * (1 * u.year)
                        else:
                            subsidies = (sf_ts * driver_ts).bin_integrals(
                                bin_boundaries=boundaries)
                        by_program_reason_pt_driver[program, reason, pt, driver] \
                                = subsidies.to(u.mega_CAD)
        self._computed_annual_subsidies = SubsidyResults(
            by_program_reason_pt_driver=by_program_reason_pt_driver)
        return self._computed_annual_subsidies

    @property
    def latest(self):
        class Latest(object):
            def __getattr__(_, attr):
                # setting this to 1e-6 with u.years gets rounded off and doesn't work
                return self.sts[attr].query(self.t_now - 1e-5 * u.seconds)
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
        if self._t_now > t_stop:
            assert 0
            return

        self._computed_annual_emissions = None
        self._computed_annual_subsidies = None

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

        if not self._heap:
            self.t_now = t_stop

Scenario = State

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


from . import ipcc_canada
from . import ghgvalues

class Other_NIR_Historical_Actuals(BaseScenarioProject):
    """Populate otherwise-missing IPCC Categories with historical actuals from NIR-2025
    """

    def on_add_project(self, state):
        non_agg_years = list(set(ipcc_canada.non_agg['Year'].unique()))
        non_agg_years.sort()
        datalen = len(non_agg_years)
        assert ('kt',) == ipcc_canada.inv['Unit'].unique()

        for pt in PT:
            driver_name = f'NIR Emissions Placeholder - {pt.value}'
            driver_sts = SparseTimeSeries(
                identifier=driver_name,
                default_value=1.0 * u.dimensionless,
                t_unit=u.year)
            state.declare_sts(self, sts=driver_sts, write=True)
            state.register_driver(
                pt=pt,
                driver='NIR Emissions Placeholder',
                sts_key=driver_name)

        # assume that these sectors, for which some dynamic element
        # has registered an emission, are considered approximate.
        registered_ipcc_sectors = {
            ipcc_sector_key
            for by_pt in state.registries['emission_factor'].values()
            for by_ipcc_sector in by_pt.values()
            for ipcc_sector_key in by_ipcc_sector}

        non_agg = ipcc_canada.inv[ipcc_canada.inv['Total'] != 'y']
        for catpathww, nonagg_catpath in non_agg.groupby('CategoryPathWithWhitespace'):
            ipcc_sector = IPCC_Sector_from_catpath_with_whitespace[catpathww]
            if ipcc_sector in registered_ipcc_sectors:
                continue

            for region, region_df in nonagg_catpath.groupby('Region'):
                if region.lower() == 'canada':
                    continue
                elif region == 'Northwest Territories and Nunavut':
                    pt = PT.XX
                else:
                    pt = PT(region)

                for ghg in GHG:
                    values = region_df[ghg.value].values
                    years = region_df['Year'].values
                    kt_by_yr = {int(year): float(val) for year, val in zip(years, values)}

                    if not all(vv == 0 for vv in kt_by_yr.values()):
                        #print(ghg, ipcc_sector, pt)
                        #print(kt_by_yr)
                        name = f'Historical {ghg.value} from {ipcc_sector.value} in {pt.value}'
                        scale = (1.0 if ghg in [GHG.CO2, GHG.CH4, GHG.N2O]
                                 else 1.0 / ghgvalues.GWP_100[ghg].magnitude)
                        state.declare_sts(
                            project=self,
                            sts=STS(
                                times=array.array('d', non_agg_years),
                                t_unit=u.years,
                                values=array.array('d', [0] + [
                                    scale * kt_by_yr.get(yr, 0)
                                    for yr in non_agg_years]),
                                v_unit=kt_by_ghg[ghg] / u.year,
                                interpolation='current'),
                            name=name,
                            write=True)
                        state.register_emission_factor(
                            pt=pt,
                            driver='NIR Emissions Placeholder',
                            sts_key=name,
                            ipcc_sector=ipcc_sector,
                            ghg=ghg)
