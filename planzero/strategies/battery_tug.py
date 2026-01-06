from .. import Project, SparseTimeSeries
from .. import ureg as u


class BatteryTugWithAuxSolarBarges(Project):
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

