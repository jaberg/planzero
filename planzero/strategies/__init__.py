import torch

from . import stakeholders
from . import battery_tug

from .. import Project, SparseTimeSeries
from .. import ureg as u

class Strategy(Project):

    may_register_emissions = False # strategies are not physical processes, they should not produce emissions

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

class ComboA(Strategy):
    def __init__(self, idea):
        super().__init__()
        self.idea = idea
        self.init_add_subprojects([
            NationalBovaerMandate(idea=stakeholders.ideas.national_bovaer_mandate),
            battery_tug.BatteryTugWithAuxSolarBarges(idea=stakeholders.ideas.battery_tugs_w_aux_solar_barges),
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


class Force_Government_ZEVs(Strategy):
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


class NationalBovaerMandate(Strategy):
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


def standard_strategies():
    return [
            ComboA(idea=stakeholders.ideas.combo_a),
            NationalBovaerMandate(idea=stakeholders.ideas.national_bovaer_mandate),
            battery_tug.BatteryTugWithAuxSolarBarges(idea=stakeholders.ideas.battery_tugs_w_aux_solar_barges),
            Force_Government_ZEVs(),
        ]
