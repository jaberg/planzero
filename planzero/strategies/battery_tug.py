from .. import SparseTimeSeries
from .. import ureg as u
from .. import mapml

from .strategy import Strategy, StrategyPage, StrategyPageSection, HTML_raw, HTML_Markdown


class BC_BatteryTug(Strategy):
    """
    The idea: use battery-powered tugs to move bulk cargo barges carrying cargo such as fuel and lumber.
    Charge batteries from grid power in the south, and from tidal power in the north.
    The financing and environmental impact model is based on a plan of
    replacing the fleet of diesel
    tugs as they age out, and then replacing old battery tugs at the same rate.
    """

    stepsize:object = 1.0 * u.years
    year_0:object = 2027 * u.years
    vessel_lifetime:object = 20 * u.years
    r_and_d_duration:object = 5 * u.years
    battery_range_time:object = 24 * u.hours * 1
    average_power_required:object = 2000 * u.horsepower  # between chargings
    non_battery_cost_per_tug:object = 5_000_000 * u.CAD
    r_and_d_cost_rate:object =  1_000_000 * u.CAD / u.year
    price_of_diesel:object = 1.20  * u.CAD / u.liter
    diesel_tug_fuel_consumption:object = 400 * u.liter / u.hour
    price_of_electricity:object = 0.1 * u.CAD / (u.kilowatt * u.hour)
    working_days_per_year:object = 300

    promising_tidal_charging_locs:list[object] = [
        (49.77, -123.95), # Skookumchuck Narrows
        (50.12, -125.35), # Discovery Passage north of Campbell River
        (49.83, -127.02), # Queens Cove, central west side Vancouver Island
        (50.46, -127.96), # Mahatta River, north west Vancouver Island
        (50.9, -127.9),
        (51.75, -127.97),
        (52.25, -128.41),
        (52.42, -128.50),
        (52.99, -129.25),
        (53.19, -129.56),
        (53.61, -130.34),
        (53.86, -130.08),
        (54.66, -130.45),
        (54.46, -130.80),
    ]

    def __init__(self):
        super().__init__(
            title="BC Battery Tugs",
            description="Replace dieself BC barge tugs with equivalent battery-powered ones, supported by tidal power generation.",
            ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
            )

    @property
    def battery_capacity_per_tug(self):
        return self.average_power_required * self.battery_range_time

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.BatteryTug_battery_cost_per_tug = SparseTimeSeries(unit=u.megaCAD)
            ctx.BatteryTug_non_battery_cost_per_tug = SparseTimeSeries(
                default_value=self.non_battery_cost_per_tug) # TODO: research

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
        # suppose everything same but fuel, so revenue is fuel savings
        fuel_cost_rate = (
            self.price_of_diesel
            * self.diesel_tug_fuel_consumption
            * (self.working_days_per_year / 365)).to(u.CAD/u.year)

        # TODO: todo import this price as a SparseTimeSeries
        electricity_cost_rate = (
            self.price_of_electricity
            * self.battery_capacity_per_tug / u.day
            * (self.working_days_per_year / 365)).to(u.CAD/u.year)
        return fuel_cost_rate - electricity_cost_rate

    def step(self, state, current):
        r_and_d_costs = 0 * u.CAD
        if state.t_now >= self.year_0:
            current.BatteryTug_r_and_d_cost_rate = self.r_and_d_cost_rate
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

    def strategy_page(self, project_comparison):
        return StrategyPage(
            show_table_of_contents=True,
            sections=[
                self.section_financial_model(project_comparison),
                self.section_tidal_power(),
                self.strategy_page_section_environmental_model(project_comparison),
            ])

    def section_financial_model(self, project_comparison):
        rval = StrategyPageSection(
            identifier='financial_section',
            title='Financial Projections',
            elements=[])

        rval.append_str_as_paragraph("""
        Suppose that the combined diesel and electric fleet size is constant.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='n_pacific_log_tugs',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        Suppose that the number of ZEV vessels increases linearly at a steady
        rate sufficient to replace the fleet every {self.vessel_lifetime}.
        Then we have the following battery-electric fleet size over time.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='n_pacific_log_tugs_ZEV',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))


        rval.append_str_as_paragraph(f"""
        The largest and riskiest factor in this strategy forecast is the projected cost of marine batteries.
        In 2026 this is a guess at the cost of Corvus Energy Blue Whale system, and looking further out,
        it is projected that the cost of marine batteries will drop via the use of Sodium Ion technology, especially from Chinese firm CATL.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='BatteryTug_battery_cost_per_tug',
                        t_unit='years',
                        figtype='plot'),
                    project_comparison.state_A,
                    project_comparison)))

        #<li>needing {self.r_and_d_annual_cost.to(u.megaCAD/u.year)} for research, development, and operations,</li>
        rval.append_str_as_paragraph(f"""
        In terms of financial modelling, the project is modelled as
        <ul>
        <li>vessels lasting {(self.vessel_lifetime).to(u.years)} with no salvage value</li>
        <li>research and development starting in {self.year_0.magnitude}, lasting {self.r_and_d_duration} and costing {self.r_and_d_cost_rate}</li>
        <li>on-board battery range sufficient for {self.battery_range_time} in average conditions, maintaining {self.average_power_required}</li>
        <li>the non-battery cost of each launched tug: {self.non_battery_cost_per_tug} </li>
        <li>diesel costing {self.price_of_diesel} and being consumed at {self.diesel_tug_fuel_consumption}</li>
        <li>electricity being available at {self.price_of_electricity}</li>
        <li>earning {(self.fuel_cost_rate * u.year).to(u.megaCAD)} annually relative to a diesel vessel in consuming less-expensive fuel</li>
        <li>vessels working {self.working_days_per_year} days per year</li>
        </ul>
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key=self.after_tax_cashflow_name,
                        t_unit='years',
                        figtype='plot'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        Although a small part of the bigger picture, the impact on CO2 emissions from the Pacific log tug fleet is expected to be significant.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='pacific_log_barge_CO2',
                        t_unit='years',
                        figtype='plot vs baseline'),
                    project_comparison.state_A,
                    project_comparison)))
        return rval

    def section_tidal_power(self):
        rval = StrategyPageSection(
            identifier='tidal_charging_section',
            title='Tidal Power for Coastal Recharging',
            elements=[])
        rval.elements.append(HTML_Markdown(content=tidal_power_markdown0))
        rval.elements.append(self.tidal_prospect_locations_viewer())
        rval.elements.append(HTML_Markdown(content=tidal_power_markdown1))
        return rval

    def tidal_prospect_locations_viewer(self):
        viewer = mapml.MapML_Viewer(
            zoom=4,
            lon=-129.0,
            lat=54.7,
            width=600,
            height=400,
            controls=True,
            layers=[
                mapml.OpenStreetMap(),
                mapml.FeatureLayer(
                    features=[
                        mapml.Point(lat=lat, lon=lon, label="")
                        for lat, lon in self.promising_tidal_charging_locs]),
                ])
        return viewer


tidal_power_markdown0 = """

British Columbia has significant tidal energy potential, estimated at
over 2,000 MW of development opportunities due to its extensive
coastline and narrow channels. [Citation TODO]
While the province was an early pioneer
in the technology, many projects remain at the demonstration or
feasibility stage rather than full-scale commercial operation.

### Notable Tidal Power Projects

* [Blind Channel Tidal Energy Demonstration Centre](https://onlineacademiccommunity.uvic.ca/primed/blind-channel/):
  Located on West Thurlow
  Island, this project is a "real-life laboratory" led by the University of
  Victoria’s Pacific Regional Institute for Marine Energy Discovery (PRIMED). It
  features a 25 kW tidal energy converter (TEC) integrated into a hybrid system
  with solar and diesel power to help remote coastal communities transition away
  from fossil fuels.
* Kamdis Tidal Power Project (Haida Gwaii): An active project involving
  [Yourbrook Energy Systems Ltd.](https://www.yourbrookenergy.com/) to develop a 500 kW tidal energy generation
  system. It combines tidal power with pumped hydroelectric storage to provide
  firm, reliable clean power to the north grid of Haida Gwaii.
* Dent Island Tidal Power Generation Project: A project developed by [Water
  Wall Turbine Inc.](https://wwturbine.com/) for the Dent Island Lodge. It utilized a floating 500 kW
  tidal turbine and battery storage system designed for shallow, narrow tidal
  areas typical of B.C.'s west coast.
* Race Rocks Tidal Energy Project: Historically significant as Canada's first
  in-stream tidal current generator, installed in 2006. The 65 kW prototype at
  the Race Rocks Ecological Reserve was used to replace diesel generators for
  several years before being decommissioned in 2011 for analysis and historical
  preservation.
* Fundy Ocean Research Centre (FORCE): a partnership of Eauclaire Tidal and
  [Orbital Marine](https://www.orbitalmarine.com/) has secured [12.5MW of marine energy licenses](https://www.orbitalmarine.com/canadian-tidal-stream-expansion/) from the Province
  of Nova Scotia for tidal stream energy deployments, which could inform larger
  scale projects in British Columbia.

### Key Locations for Tidal Resources

* Discovery Passage: Areas near Campbell River are frequently studied due to
  fast currents that offer easy access to the BC Hydro power grid.
* Skookumchuck Narrows (Sechelt Rapids): Known for some of the world's fastest
  tidal currents, reaching speeds up to 17.6 knots.
* Haida Gwaii & Vancouver Island: These regions host numerous off-grid
  communities where tidal power is viewed as a critical alternative to diesel
  dependency.

Simply looking along aerial photography of the BC coast suggests a number of
narrow-mouthed inlets which might be candidates for further study of power generation
potential.
"""

tidal_power_markdown1 = """
### Regulatory and Development Context

The B.C. government manages Ocean Energy Tenures on Crown land through
investigative and general area licenses. While the province has vast
resources, recent development has faced challenges due to the high cost of
technology compared to B.C.'s established hydroelectric system and a difficult
regulatory climate for nascent marine renewables.
"""




"https://www.seaspan.com/stories/log-barging-101/",
