import enum
from .my_functools import cache
import matplotlib.pyplot as plt

from .enums import PT, GHG, IPCC_Sector as IPCC
from .est_nir_util import _rstrip_data, _echart_reference_NIR_values
from .ghgvalues import GWP_100
from .html import (
    EChartTitle,
    EChartXAxis,
    EChartYAxis,
    EChartSeriesStackElem,
    EChartSeriesBase,
    EChartLineStyle,
    EChartItemStyle,
    StackedAreaEChart)
from .nrc_nfd import (
    RoundwoodProductCategory,
    RoundwoodTenure,
    RoundwoodSpeciesGroup,
    m3_by_roundwood_species_group,
    net_merchantable_volume_harvested
    )
from .ureg import (
    u,
    kt_by_ghg,
    )
from . import eccc_nir_annex6
from . import ipcc_canada
from . import objtensor
from . import sts

RPC = RoundwoodProductCategory

C_coef = .5 * u.kg_carbon / u.kg_wood_od # Annex6 Table 5-1
CO2_coef = 3.667 * u.kg_CO2 / u.kg_carbon # molecule weight ratio

decay_horizon = 2025 * u.years

_echart_years = ipcc_canada.echart_years()

def HWP_zeros():
    rval = objtensor.empty(GHG, RoundwoodProductCategory, PT)
    for ghg, kt in kt_by_ghg.items():
        rval[ghg] = 0 * kt
    return rval


def _delayed_CO2_released(CO2_by_pt, rpc, halflife):
    rval = HWP_zeros()
    for pt in PT:
        report = CO2_by_pt[pt]
        if isinstance(report, sts.STS):
            report = sts.annual_report_decay(
                report, halflife * u.years, decay_horizon)
        rval[GHG.CO2, rpc, pt] = report
    return rval


@cache
def _logs_and_bolts():
    # The carbon content of net harvested timber is accounted as
    # being instantaneously decremented from the forest (considered as part of the atmosphere) and slowly
    # incremented in the HWP sector. This is physically
    # unrealistic in that the CO2 was absorbed slowly into the
    # tree over the tree's lifetime, not all at once.
    # sector. But the HWP increment
    #
    # Gemini suggests "Mill Efficiency Factor" of 50-60%
    # ends up as long-term lumber, and the figure can be
    # found in NIR. The rest will be end up combusted (though
    # possibly some paper?).

    net_harvested_w_tenure = net_merchantable_volume_harvested() # cache'd
    net_harvested = net_harvested_w_tenure.sum(RoundwoodTenure)

    idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
        'Species-weighted average density, Sawnwood')
    assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
    density_magnitude = eccc_nir_annex6.data_a6_5_2['Value'][idx]
    mill_efficiency = .7

    SG = RoundwoodSpeciesGroup
    CO2_by_sg = objtensor.empty(SG, PT)
    for sg in SG:
        avg_density = (
            density_magnitude
            * (u.tonne_wood_od / m3_by_roundwood_species_group[sg]))
        CO2_by_sg[sg] = (
            net_harvested[sg, RPC.Logs_and_Bolts]
            * (mill_efficiency * avg_density * C_coef * CO2_coef))
        # TODO: What about N2O and CH4 - do these products go to landfill?

    # will be deducted from Forest_Land
    CO2_by_pt = CO2_by_sg.sum(SG)
    cap = HWP_zeros()
    cap[GHG.CO2, RPC.Logs_and_Bolts] = CO2_by_pt
    rel = _delayed_CO2_released(
        CO2_by_pt, RPC.Logs_and_Bolts,
        halflife=eccc_nir_annex6.dict_a6_5_3['Canada', 'Sawnwood'])
    return cap, rel


@cache
def _other_industrial_roundwood():
    net_harvested_w_tenure = net_merchantable_volume_harvested() # cache'd
    net_harvested = net_harvested_w_tenure.sum(RoundwoodTenure)

    # pilings, railway ties, electricity poles
    # similar to Logs and Bolts but lasts slightly longer
    idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
        'Species-weighted average density, Other industrial roundwood')
    assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
    density_magnitude = eccc_nir_annex6.data_a6_5_2['Value'][idx]
    mill_efficiency = .7

    SG = RoundwoodSpeciesGroup
    CO2_by_sg = objtensor.empty(SG, PT)
    for sg in SG:
        avg_density = (
            density_magnitude
            * (u.tonne_wood_od / m3_by_roundwood_species_group[sg]))
        CO2_by_sg[sg] = (
            net_harvested[sg, RPC.Other_Industrial_Roundwood]
            * (mill_efficiency * avg_density * C_coef * CO2_coef))
        # What about N2O and CH4 - do these products go to landfill?

    # will be deducted from Forest_Land
    CO2_by_pt = CO2_by_sg.sum(SG)
    cap = HWP_zeros()
    cap[GHG.CO2, RPC.Other_Industrial_Roundwood] = CO2_by_pt
    rel = _delayed_CO2_released(
        CO2_by_pt, RPC.Other_Industrial_Roundwood,
        halflife=eccc_nir_annex6.dict_a6_5_3['Canada', 'Other industrial roundwood'])
    return cap, rel


@cache
def _fuelwood_and_firewood():
    net_harvested_w_tenure = net_merchantable_volume_harvested() # cache'd
    net_harvested = net_harvested_w_tenure.sum(RoundwoodTenure)

    # Fuelwood CO2 is considered to released
    # back to the atmosphere from whence it came
    # but N2O and CH4 are considered new emissions
    idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
        'Species-weighted average density, Bioenergy')
    assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
    density_raw = eccc_nir_annex6.data_a6_5_2['Value'][idx]

    cap = HWP_zeros()
    rel = HWP_zeros()

    SG = RoundwoodSpeciesGroup
    for sg in SG:
        # extra 15% moisture content, supposing oven-dried is 10% moisture
        # content, and mc25 is, by definition, 25.
        avg_density = (
            (density_raw + .15)
            * (u.tonne_wood_mc25 / m3_by_roundwood_species_group[sg]))

        mass_wood_mc25 = net_harvested[sg, RPC.Fuelwood_and_Firewood] * avg_density
        # eyeballing table Annex6 6-1
        # using coefficients for residential combustion
        CH4_coef = 5.0 * u.g_CH4 / u.kg_wood_mc25
        N2O_coef = 0.06 * u.g_N2O / u.kg_wood_mc25
        rel[GHG.CH4, RPC.Fuelwood_and_Firewood] += CH4_coef * mass_wood_mc25
        rel[GHG.N2O, RPC.Fuelwood_and_Firewood] += N2O_coef * mass_wood_mc25
    return cap, rel


@cache
def _pulpwood():
    net_harvested_w_tenure = net_merchantable_volume_harvested() # cache'd
    net_harvested = net_harvested_w_tenure.sum(RoundwoodTenure)

    # Pulpwood C is considered to be removed from
    # forestry sector when it's harvested, but then
    # returned to the atmosphere over the following years
    idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
        'Species-weighted average density, Roundwood')
    assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
    density_magnitude = eccc_nir_annex6.data_a6_5_2['Value'][idx]

    SG = RoundwoodSpeciesGroup
    CO2_by_sg = objtensor.empty(SG, PT)
    for sg in SG:
        avg_density = (
            density_magnitude
            * (u.tonne_wood_od / m3_by_roundwood_species_group[sg]))
        CO2_by_sg[sg] = (
            net_harvested[sg, RPC.Pulpwood]
            * (avg_density * C_coef * CO2_coef))

        # The CH4 and N2O are supposed to be picked up by Waste sectors,
        # such as landfill. TODO: figure out a lower bound on landfill
        # emissions?

    CO2_by_pt = CO2_by_sg.sum(SG)
    cap = HWP_zeros()
    cap[GHG.CO2, RPC.Pulpwood] = CO2_by_pt
    rel = _delayed_CO2_released(
        CO2_by_pt, RPC.Pulpwood,
        halflife=eccc_nir_annex6.dict_a6_5_3['Canada', 'Pulp and paper'])

    return cap, rel


class EstForestAndHarvestedWoodProducts(object):
    """
    Canada's forests are modelled here as constant-size, constant-composition
    ecosystems, which buffer atmospheric carbon. This model ignores types and
    location of forest, although it does recognize the different density of
    softwood and hardwood species groups.

    The model is that the tree population absorb carbon from the atmosphere at
    a constant rate, and releases it at a variable rate in three ways:

    1. harvesting for durable products (transfers C to products & atmosphere)
    2. decay after insect infestation
    3. combustion due to wild fire

    Data supporting this simple model is drawn from 
    http://nfdp.ccfm.org/en/download.php

    """


    def __init__(self):
        # This initial value doesn't really matter, it isn't trying
        # to reflect a physical quantity.

        # This rate of accumulation does matter, the intention is to
        # approximately match the trend of forestry sector emissions reported
        # by the ECCC. This is not a very interesting model, in that it
        # doesn't attempt to reveal anything about why the trend is the value
        # that it is. NRCan does world-class bottom-up forest carbon
        # modelling, including with open-source software, although I couldn't
        # yet find public data for that software. It would be interesting
        # future work to use that model instead.
        #self.reforestation_rate = 50 * u.megatonne_wood_mc25 / u.year

        self.net_harvested_w_tenure = net_merchantable_volume_harvested()
        # ignore tenure
        self.net_harvested = self.net_harvested_w_tenure.sum(RoundwoodTenure)


        # how much of each GHG is transferred from Forest_Land -> HWP
        self.HWP_captured = HWP_zeros()

        # how much of each GHG is transferred from HWP -> Atmosphere
        self.HWP_released = HWP_zeros()

        for cat in RPC:
            if cat == RPC.Logs_and_Bolts:
                cap, rel = _logs_and_bolts()
            elif cat == RPC.Other_Industrial_Roundwood:
                cap, rel = _other_industrial_roundwood()
            elif cat == RPC.Fuelwood_and_Firewood:
                cap, rel = _fuelwood_and_firewood()
            elif cat == RPC.Pulpwood:
                cap, rel = _pulpwood()
            else:
                raise NotImplementedError(cat)
            self.HWP_captured += cap
            self.HWP_released += rel

        self.forest_emissions = self.HWP_captured.sum(RPC)
        self.HWP_emissions = self.HWP_released.sum(RPC) - self.forest_emissions

    def plot_forest_emissions(self):
        ptvalues.scatter_subplots(
            self.forest_emissions[[GHG.CO2, GHG.CH4, GHG.N2O]],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='upper left')

    def plot_HWP_emissions(self):
        ptvalues.scatter_subplots(
            self.HWP_emissions[[GHG.CO2, GHG.CH4, GHG.N2O]],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='upper left')

    def plot_total_co2e(self):
        a9_totals = {
            year: eccc_nir_annex9.emissions_by_IPCC_sector(year, 'Total_CO2e')
            for year in range(1990, 2020)}

        fig, (ax0, ax1) = plt.subplots(1, 2)

        forest = (GWP_100 @ self.forest_emissions.sum(PT)).to(u.kt_CO2e)
        ax0.scatter(
            forest.times,
            forest.values[1:],
            label='Estimate')
        ax0.scatter(
            [int(yr) for yr in forest.times if yr in a9_totals],
            [a9_totals[yr][IPCC.Forest_Land].to(u.kt_CO2e).magnitude
             for yr in forest.times if yr in a9_totals],
            label='NIR')
        ax0.set_title('Forest')

        hwp = (GWP_100 @ self.HWP_emissions.sum(PT)).to(u.kt_CO2e)
        ax1.scatter(
            hwp.times,
            hwp.values[1:],
            label='Estimate')
        ax1.scatter(
            [int(yr) for yr in hwp.times if yr in a9_totals],
            [a9_totals[int(yr)][IPCC.Harvested_Wood_Products].to(u.kt_CO2e).magnitude
             for yr in hwp.times if yr in a9_totals],
            label='NIR')
        ax1.set_title('HWP')

    def update_A9_emissions(self, sectoral_emissions):
        sectoral_emissions[:, IPCC.Forest_Land] \
                += self.forest_emissions
        sectoral_emissions[:, IPCC.Harvested_Wood_Products] \
                += self.HWP_emissions

    def echart_forest_land(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Forest Land')

        forest_co2e = GWP_100 @ self.forest_emissions.sum(PT)

        return StackedAreaEChart(
            div_id='ipcc_chart_forest_land',
            title=EChartTitle(
                text='Emissions from Forest Land',
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name='Harvested Wood Products',
                    data=_rstrip_data(forest_co2e.to(u.Mt_CO2e))),
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
            ])

    def echart_HWP(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Harvested Wood Products')

        ts_dict = {
            (rpc, 'captured'): -GWP_100 @ self.HWP_captured[:, rpc].sum(PT)
            for rpc in RPC if rpc != RPC.Fuelwood_and_Firewood}
        ts_dict.update({
            (rpc, 'released'): GWP_100 @ self.HWP_released[:, rpc].sum(PT)
            for rpc in RPC if True or rpc != RPC.Fuelwood_and_Firewood})

        net_emissions = GWP_100 @ self.HWP_emissions.sum(PT)

        return StackedAreaEChart(
            div_id='ipcc_chart_hwp',
            title=EChartTitle(
                text='Emissions from Harvested Wood Products',
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=f'{rpc.value} {cap_or_rel}',
                    data=_rstrip_data(ts.to(u.Mt_CO2e)))
                for (rpc, cap_or_rel), ts in ts_dict.items()],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
                EChartSeriesBase(
                    name='Net estimate',
                    lineStyle=EChartLineStyle(color='#606060'),
                    itemStyle=EChartItemStyle(color='#606060'),
                    data=_rstrip_data(net_emissions.to(u.Mt_CO2e))),
            ])

    def echart_HWP_and_forest_land_ref(self):
        non_agg = ipcc_canada.non_agg
        years = ipcc_canada.echart_years()
        forest_land_values = non_agg[non_agg['CategoryPathWithWhitespace'] == 'Forest Land']['CO2eq'].values / 1000
        HWP_values = non_agg[non_agg['CategoryPathWithWhitespace'] == 'Harvested Wood Products']['CO2eq'].values / 1000

        return StackedAreaEChart(
            div_id='ipcc_chart_HWP_forest_land_ref',
            title=EChartTitle(
                text='Comparing Emissions from Forest Land and HWP',
                subtext=''),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[],
            other_series=[
                EChartSeriesBase(
                    name='Forest Land (from NIR)',
                    lineStyle=EChartLineStyle(color='#303030', type='dashed'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=forest_land_values),
                EChartSeriesBase(
                    name='Harvested Wood Products (from NIR)',
                    lineStyle=EChartLineStyle(color='#303030', type='dotted'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=HWP_values),
                EChartSeriesBase(
                    name='Combined',
                    lineStyle=EChartLineStyle(color='#303030', width=3),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=HWP_values + forest_land_values),
            ],
            legend={"top": 40})


