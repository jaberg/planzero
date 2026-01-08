from .. import Project, SparseTimeSeries
from .. import ureg as u
from .ideas import ideas


class BatteryFreighter(Project):
    """
    A Battery Freighter is contemplated for operation on the Great Lakes, which
    is a battery version of modern Equinox-class vessels developed by Algoma.

    The idea is that such vessels would be charged by grid electricity at some locations,
    and at other locations by anchored floating or shoreline solar arrays
    (with perhaps co-located wind or diesel generators).

    A vessel is contemplated that would charge once per day, and that the
    charging locations would be able to charge one ship per day.

    """

    battery_cost_per_capacity = 200 * u.CAD / (1280 * u.watt * u.hour)
    battery_energy_density = 150 * u.watt * u.hour / u.kg

    def __init__(self, year_0=2026 * u.years):
        super().__init__()
        self.idea = ideas.battery_freighter
        self.stepsize = 1.0 * u.years
        self.after_tax_cashflow_name = f'{self.__class__.__name__}_AfterTaxCashFlow'
        self.year_0 = year_0
        self.vessel_lifetime = 20 * u.years
        self.r_and_d_duration = 5 * u.years
        self.battery_range_time = 24 * u.hours * 1
        self.target_fleet_size = 30

        self.power_required_per_working_day = 5000 * u.horsepower  # average power over worst-case working day

        self.battery_capacity = self.power_required_per_working_day * self.battery_range_time

        self.battery_cost = (self.battery_capacity * self.battery_cost_per_capacity).to(u.CAD)
        self.battery_mass = self.battery_capacity / self.battery_energy_density
        print(self.battery_cost, self.battery_mass.to(u.tonne))
        self.non_battery_cost = 20_000_000 * u.CAD # guess

        self.cost = self.battery_cost + self.non_battery_cost

        self.r_and_d_annual_cost = 1_000_000 * u.CAD / u.year

        self.pre_battery_deadweight = 39_000 * u.tonne
        self.deadweight = self.pre_battery_deadweight - self.battery_mass


    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.n_great_lakes_available_battery_freighters = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            # two reasons this is a timeseries
            # 1. so that GreatLakesFreight can see values we set in this class
            # 2. so that the value can change over time if we assume battery improvements
            ctx.average_battery_freighter_capacity = SparseTimeSeries(
                default_value=0 * u.tonne)

            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now


    @property
    def fuel_cost_rate(self):
        price_of_diesel = 1.20  * u.CAD / u.liter
        working_days_per_year = 300

        # TODO: how does this compare with Equinox-class efficiency, which is
        # supposed to be the baseline.
        diesel_tug_fuel_consumption = (
            400 * u.liter / u.hour  # imagine same as tug
            * self.power_required_per_working_day / (2000.0 * u.horsepower)) # but for a larger engine
        # suppose everything same but fuel, so revenue is fuel savings
        fuel_cost_rate = (
            price_of_diesel
            * diesel_tug_fuel_consumption
            * (working_days_per_year / 365)).to(u.CAD/u.year)

        # TODO: work out actual electricity cost under some plan to provide
        # the electricity.
        # TODO: use average day electricity demand, not the whole battery capacity
        price_of_electricity = 0.1 * u.CAD / (u.kilowatt * u.hour)
        electricity_cost_rate = (
            price_of_electricity
            * self.battery_capacity / u.day
            * (working_days_per_year / 365)).to(u.CAD/u.year)
        return fuel_cost_rate - electricity_cost_rate

    @property
    def tug_labour_rate(self):
        crew_size = 16
        avg_salary = 75_000 * u.CAD / u.year # total guess, no idea
        cost_rate = crew_size * avg_salary
        return cost_rate

    def step(self, state, current):
        current.average_battery_freighter_capacity = self.deadweight

        if state.t_now >= self.year_0:
            r_and_d_costs = self.r_and_d_annual_cost * self.stepsize
        else:
            r_and_d_costs = 0 * u.CAD


        if state.t_now >= self.year_0 + self.r_and_d_duration:
            n_freighters_built_per_step = (
                self.target_fleet_size
                / self.vessel_lifetime) * self.stepsize
            if state.latest.n_great_lakes_available_battery_freighters < self.target_fleet_size:
                current.n_great_lakes_available_battery_freighters += n_freighters_built_per_step
            else:
                # we don't model the explicit decomissioning and recommisioning every 20 years
                pass
            manufacturing_costs = n_freighters_built_per_step * (
                self.cost)
        else:
            manufacturing_costs = 0 * u.CAD

        # differential operating revenue vs. conventional freighter
        revenue = state.latest.n_great_lakes_active_battery_freighters * self.fuel_cost_rate * self.stepsize
        
        # not included:
        # salvage value, part-reuse between vessel generations (same?)
        # maintenance (same?)
        # insurance (same?)
        # financing (more)
        # port fees (presumably same)

        setattr(
            current,
            self.after_tax_cashflow_name, 
            - r_and_d_costs
            - manufacturing_costs
            + revenue
            )
        return state.t_now + self.stepsize

    def project_page_vision(self): 
        return self.__class__.__doc__

    def project_page_graphs(self):
        rval = []

        descr = f"""
        Suppose that the number of ZEV vessels increases linearly to a target fleet size of {self.target_fleet_size}
        at a steady
        rate sufficient to replace the fleet every {self.vessel_lifetime}.
        """
        rval.append(dict(
            sts_key='n_great_lakes_available_battery_freighters',
            t_unit='years',
            descr=descr))

        descr = """
        The impact on CO2 emissions from the Great Lakes fleet is expected to be significant.
        """
        rval.append(dict(
            sts_key='great_lakes_freight_CO2',
            t_unit='years',
            figtype='plot vs baseline',
            descr=descr))

        descr = """
        The impact on atmospheric CO2 concentration is expected to be very small.
        That said, if the technology used in this solution were scaled globally,
        the impact could be larger. (TODO: estimate how much larger)
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
        <li>needing {self.cost.to(u.megaCAD)} per vessel at a rate sufficient to replace the fleet over vessel lifetime of {self.vessel_lifetime}
        <li>earning {(self.fuel_cost_rate * u.year).to(u.megaCAD)} annually per vessel [set] in saved fuel</li>
        </ul>
        """
        rval.append(dict(
            sts_key=self.after_tax_cashflow_name,
            t_unit='years',
            figtype='plot',
            descr=descr))
        return rval

