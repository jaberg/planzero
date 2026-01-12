from .. import Project, SparseTimeSeries
from .. import ureg as u


class BatteryTug(Project):
    """
    Battery Tug strategy is to develop a tug as a replacement for BC Log Barge-towing tugs.

    It includes the construction of wind-diesel hybrid generators powered
    recharging stations along the BC coast.
    The recharging stations are meant to mainly use wind, and use diesel
    to make up for (1) wind variability and (2) demand variability.
    """

    battery_cost_per_capacity = 200 * u.CAD / (1280 * u.watt * u.hour)
    # TODO: cost multiplier for marine-grade batteries
    # TODO: use state.latest.LithiumIonBatteryTechnology_cost

    def __init__(self, idea=None, year_0=2026 * u.years):
        super().__init__()
        self.idea = idea
        self.stepsize = 1.0 * u.years
        self.after_tax_cashflow_name = f'{self.__class__.__name__}_AfterTaxCashFlow'
        self.year_0 = year_0
        self.vessel_lifetime = 20 * u.years
        self.r_and_d_duration = 5 * u.years
        self.battery_range_time = 24 * u.hours * 1

        # Application constants that are not expected to change over time
        self.average_power_required = 2000 * u.horsepower  # between chargings
        self.battery_capacity_per_tug = self.average_power_required * self.battery_range_time

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.BatteryTug_battery_cost_per_tug = SparseTimeSeries(unit=u.megaCAD)
            ctx.BatteryTug_non_battery_cost_per_tug = SparseTimeSeries(
                default_value=5_000_000 * u.CAD) # TODO: research

            ctx.BatteryTug_cost_per_tug = SparseTimeSeries(unit=u.CAD)
            ctx.BatteryTug_r_and_d_cost_rate = SparseTimeSeries(
                default_value=0 * u.CAD / u.year)
            ctx.n_pacific_log_tugs_ZEV_constructed = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    @property
    def fuel_cost_rate(self):
        # TODO: todo import this price as a SparseTimeSeries
        price_of_diesel = 1.20  * u.CAD / u.liter
        working_days_per_year = 300
        diesel_tug_fuel_consumption = 400 * u.liter / u.hour
        # suppose everything same but fuel, so revenue is fuel savings
        fuel_cost_rate = (
            price_of_diesel
            * diesel_tug_fuel_consumption
            * (working_days_per_year / 365)).to(u.CAD/u.year)

        # TODO: todo import this price as a SparseTimeSeries
        price_of_electricity = 0.1 * u.CAD / (u.kilowatt * u.hour)
        electricity_cost_rate = (
            price_of_electricity
            * self.battery_capacity_per_tug / u.day
            * (working_days_per_year / 365)).to(u.CAD/u.year)
        return fuel_cost_rate - electricity_cost_rate

    def step(self, state, current):
        r_and_d_costs = 0 * u.CAD
        if state.t_now >= self.year_0:
            current.BatteryTug_r_and_d_cost_rate = 1_000_000 * u.CAD / u.year
            r_and_d_costs += current.BatteryTug_r_and_d_cost_rate * self.stepsize

        manufacturing_costs = 0 * u.CAD
        if state.t_now >= self.year_0 + self.r_and_d_duration:
            n_tugs_built_per_step = (
                state.latest.n_pacific_log_tugs
                / self.vessel_lifetime) * self.stepsize

            current.BatteryTug_battery_cost_per_tug = (
                state.latest.MarineBatteryTechnology_system_cost
                * self.battery_capacity_per_tug)
            current.BatteryTug_cost_per_tug = (
                current.BatteryTug_battery_cost_per_tug
                + current.BatteryTug_non_battery_cost_per_tug)

            manufacturing_costs += n_tugs_built_per_step * (
                current.BatteryTug_cost_per_tug)

            # TODO: rename this to "active" rather than "constructed" because
            # we keep constructing more and more, it's just that we
            # decommission the old ones
            if state.latest.n_pacific_log_tugs_ZEV_constructed < state.latest.n_pacific_log_tugs:
                current.n_pacific_log_tugs_ZEV_constructed += n_tugs_built_per_step
            else:
                # TODO: model the explicit decomissioning and recommisioning every 20 years
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
        tugs pulling barges of e.g. logs and fuel between Vancouver Island and Vancouver, and up and down the BC Coast.


        """
        unused_text = """
        I believe that this vessel should not be assessed for either the Great Lakes or northern supply routes, but for different reasons.
        The Great Lakes routes are connected by many locks, and are currently served by larger self-propelled vessels rather than tug-barge combinations;
        I believe the need to design and deploy barges as well as tugs as well as the larger size of the freight application motivates the design of custom vessels, although similar battery-powered logic probably applies.
        In the case of Northern resupply, I believe there is insufficient
        charging infrastructure along the routes, and the trips themselves are
        not frequent enough to motivate installing charging infrastructure. In
        both cases, a different vessel design is likely necessary.
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
        rate sufficient to replace the fleet every {self.vessel_lifetime}.
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
        In terms of financial modelling, the project is dominated by the projected cost of marine batteries.
        In 2026 this is a guess at the cost of Corvus Energy Blue Whale system, and looking further out,
        it is projected that the cost of marine batteries will drop via the use of Sodium Ion technology, especially from Chinese firm CATL.
        """
        rval.append(dict(
            sts_key='BatteryTug_battery_cost_per_tug',
            t_unit='years',
            figtype='plot',
            descr=descr))

        #<li>needing {self.r_and_d_annual_cost.to(u.megaCAD/u.year)} for research, development, and operations,</li>
        descr = f"""
        In terms of financial modelling, the project is modelled as
        <ul>
        <li>earning {(self.fuel_cost_rate * u.year).to(u.megaCAD)} annually per vessel [set] in saved fuel</li>
        </ul>
        """
        rval.append(dict(
            sts_key=self.after_tax_cashflow_name,
            t_unit='years',
            figtype='plot',
            descr=descr))
        return rval

