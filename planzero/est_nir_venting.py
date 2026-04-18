import array
from .my_functools import cache
from .naics import NAICS6
from .ureg import u, kt_by_ghg
from . import aer
from . import enums
from . import ghgrp
from .ghgvalues import GWP_100, GHG_zero_kg
from . import ipcc_canada
from . import objtensor
from . import sts
from .est_nir_util import (
    _rstrip_data,
    _echart_years,
    _echart_reference_NIR_values,
    )
from .html import (
    EChartTitle,
    EChartXAxis,
    EChartYAxis,
    EChartSeriesStackElem,
    EChartSeriesBase,
    EChartLineStyle,
    EChartItemStyle,
    StackedAreaEChart)

GHG = enums.GHG
IPCC = enums.IPCC_Sector
PT = enums.PT

def GHG_PT_zeros():
    rval = objtensor.empty(GHG, PT)
    rval[:] = GHG_zero_kg()[:, None]
    return rval

@cache
def from_ghgrp():
    nse = ghgrp.GHG_NAICS_source_emissions_backfilled()

    oil_and_gas = (
        NAICS6.Oil_and_gas_extraction__except_oil_sands,
        NAICS6.Conventional_Oil_and_Gas_Extraction,
        NAICS6.NonConventional_Oil_Extraction,
        NAICS6.Insitu_oil_sands_extraction,
        NAICS6.Mined_oil_sands_extraction,
        NAICS6.Pipeline_Transportation_of_Crude_Oil,
        NAICS6.Pipeline_Transportation_of_Natural_Gas,
    )
    EmissionSource = ghgrp.EmissionSource
    arguably_venting = (
        EmissionSource.Waste,
        EmissionSource.Venting,
        EmissionSource.IndustrialProcess,
        EmissionSource.Wastewater,
    )
    rval = nse[:, oil_and_gas, arguably_venting].sum(2).sum(1)
    return rval


@cache
def from_st60b():
    # st60b is the report on emissions from Alberta facilities that are not in the GHGRP
    st60b = aer.st60b_2024_OneStop() # volumes
    avg_methane_content_of_vented_gas = .85
    methane_mass_per_volume = (
        .6785
        * avg_methane_content_of_vented_gas
        * u.kg_CH4 / u.m3_NG_nmk)
    ch4_st60b = st60b * methane_mass_per_volume

    rval = {}
    for vt in aer.VentingType:
        label = f'{vt.value} (Alberta, ST60B report)'
        ch4_vt = ch4_st60b[vt].to(u.kt_CH4)
        # backfill to 2004 using 2020 values just to make the 2005
        # estimate less bad
        assert ch4_vt.times[0] == 2020
        new_times = list(range(2004, 2020))
        ch4_vt.times[0:0] = array.array('d', new_times)
        ch4_vt.values[1:1] = array.array('d', [ch4_vt.values[1]] * len(new_times))
        rval[label] = ch4_vt
    return rval

class EstFugitive_OilandNaturalGas_Venting(object):


    def init_petrinex_SK(self):
        from planzero.petrinex import (
            ActivityID,
            ProductID,
            FacilityType,
            petrinex_annual_summary)
        self.SK_venting_products = (
            ProductID.MethaneMix,
            ProductID.AcidGas,
            ProductID.CO2,
            ProductID.CO2Mix,
            ProductID.Condensate,
            ProductID.EntrainedGas,
            ProductID.Gas,
            ProductID.CrudeOil)

        factors = objtensor.empty(enums.GHG, self.SK_venting_products)
        for ghg in enums.GHG:
            factors[ghg] = 0 * kt_by_ghg[ghg] / u.m3

        factors[GHG.CH4, ProductID.MethaneMix] = (
            .440295 * 1000 # convert liquid methane to equivalent volume worth of gas at standard pressure
            * .6785 * u.kg_CH4 / u.m3) # mass at standard pressure

        factors[GHG.CO2, ProductID.AcidGas] = (
            .5 # Gemini thinks the molar fraction of CO2 in acid gas ranges from .05 to .95 (!?)
            * 1.861 * u.kg_CO2 / u.m3)

        factors[GHG.CO2, ProductID.CO2] = (
            1.861 * u.kg_CO2 / u.m3)

        factors[GHG.CO2, ProductID.CO2Mix] = (
            .440295 * 1000 # convert liquid CO2 to gas at standard pressure
            * 1.861 * u.kg_CO2 / u.m3)

        factors[GHG.CO2, ProductID.Condensate] = (
            .370213 * 1000 # Gemini suggested this number for how many m3 of standard pressure gas would flash out of a m3 of liquid
            * .03 # a guess at the molar fraction of CO2 in the flashed gas
            * 1.861 * u.kg_CO2 / u.m3) # mass at standard pressure

        factors[GHG.CH4, ProductID.Condensate] = (
            .370213 * 1000 # Gemini suggested this number for how many m3 of standard pressure gas would flash out of a m3 of liquid
            * .6 # a guess at the molar fraction of methane in the flashed gas
            * .6785 * u.kg_CH4 / u.m3) # mass at standard pressure

        factors[GHG.CH4, ProductID.Gas] = (
            .9 # a guess at the molar fraction of methane in the gas
            * .6785 * u.kg_CH4 / u.m3) # mass at standard pressure

        factors[GHG.CO2, ProductID.EntrainedGas] = (
            .1 # a guess at the molar fraction of CO2 in the gas
            * 1.861 * u.kg_CO2 / u.m3) # mass at standard pressure

        factors[GHG.CH4, ProductID.EntrainedGas] = (
            .8 # a guess at the molar fraction of methane in the gas
            * .6785 * u.kg_CH4 / u.m3) # mass at standard pressure

        factors[GHG.CO2, ProductID.CrudeOil] = (
            .380780 * 1000 # Gemini suggested this number for how many m3 of standard pressure gas would flash out of a m3 of liquid
            * .03 # a guess at the molar fraction of CO2 in the flashed gas
            * 1.861 * u.kg_CO2 / u.m3) # mass at standard pressure

        factors[GHG.CH4, ProductID.CrudeOil] = (
            .380780 * 1000 # Gemini suggested this number for how many m3 of standard pressure gas would flash out of a m3 of liquid
            * .6 # a guess at the molar fraction of methane in the flashed gas
            * .6785 * u.kg_CH4 / u.m3) # mass at standard pressure

        self.petrinex_SK = petrinex_annual_summary(
            PT.SK, include_ghgrp=False).sum(FacilityType)
        # I don't really know what all the activities mean
        # but Vent may be the only emission type for the IPCC Venting category
        self.vSK = self.petrinex_SK[self.SK_venting_products, ActivityID.Vent]
        emissions = GHG_PT_zeros()
        emissions[:, PT.SK] = (factors * self.vSK).sum(1)
        self.emissions_by_label['Petrinex Vent (Saskatchewan)'] = emissions

    def init_abandoned_modelling_gap(self):
        # I'm working through sectors on a time-boxed basis, and the data at time of writing
        # only got me about 2/7 of the way to the 2005 target, and 1/2 of the
        # way to the 2023 target.
        #
        # I've added this stop-gap primarily so that the
        # print_sectoral_emissions_gaps function, which is currently guiding
        # my selection of which IPCC sector to work on, doesn't immediately
        # tell me to work on this one again.
        #
        # I chose the numbers by first making the sum of emissions in this
        # class approximately match the NIR sectoral target, and then
        # subtracting a constant amount so that the modelling gap ends at 0 in
        # the latest year, 2023.  After the subtraction, the result is
        # a relatively consistent gap of about 20 Mt over the timeseries.
        #
        # Semantically, it corresponds to "fugitive emissions for which there
        # isn't data in planzero, which appear to have been addressed by 2023"

        # just eyeballing the missing data
        emission = GHG_PT_zeros()
        emission[GHG.CH4, PT.AB] = sts.annual_report2(
            years=[
                1990, 1997, 2003, 2004, 2015, 2023],
            values=[(vv - 20) / 28 for vv in [
                40,   67,   67,   48,   50,   20]],
            v_unit=u.Mt_CH4,
            ).interp(times=[tt * u.years for tt in range(1990, 2024)],)
        self.emissions_by_label['Historical modelling gap'] = emission

    def __init__(self, init=True):
        self.emissions_by_label = {}
        self.years = ipcc_canada.echart_years()
        if not init:
            return
        for label, ch4_vt in from_st60b().items():
            emissions = GHG_PT_zeros()
            emissions[GHG.CH4, PT.AB] = sts.with_default_zero(
                ch4_vt, self.years * u.years)
            self.emissions_by_label[label] = emissions
        self.emissions_by_label['Registered facilities (GHGRP)'] = from_ghgrp()
        self.init_petrinex_SK()

        # This multiplication is justified by
        # (a) The 2017 paper by Johnson et al. that said gov estimates of the
        # day (based on the *kinds* of data we're using here) underestimated
        # emissions by a factor of 2.5 (1.5 more than estimated)
        # (b) The fact that the estimator is only getting about 40% of the NIR total
        unreported = GHG_PT_zeros()

        for label in self.emissions_by_label:
            for key, off in self.emissions_by_label[label].ravel_keys_offsets():
                ghg, pt = key
                val = self.emissions_by_label[label].buf[off]
                if isinstance(val, sts.STS):
                    val.setdefault_zero([yy * u.years for yy in self.years])
                    unreported[ghg, pt] += val * 1.5

        self.emissions_by_label['Estimated Unreported'] = unreported

        self.init_abandoned_modelling_gap()

    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.Fugitive__Venting, :] += emissions

    def echart_venting(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Fugitive Sources/Oil and Natural Gas/Venting')

        return StackedAreaEChart(
            div_id='ipcc_chart_venting',
            title=EChartTitle(
                text='Emissions from Venting (Fugitive Sources / Oil and Natural Gas)',
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=label,
                    data=_rstrip_data((GWP_100 @ emissions).sum(PT).to(u.Mt_CO2e)))
                for label, emissions in self.emissions_by_label.items()
                if label != 'Historical modelling gap'
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
            ])
