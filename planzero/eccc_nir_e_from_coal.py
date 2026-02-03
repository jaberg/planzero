import matplotlib.pyplot as plt

from . import eccc_nir_annex13
from .enums import Geo, CoalType
from .ptvalues import PTValues
from .sts import annual_report, SparseTimeSeries
from .ureg import u
from .planet_model import CO2e_from_emissions
from .sc_nir import Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility


class A6_1_10_CanadianBituminous(PTValues):
    def __init__(self):
        row_1_2 = SparseTimeSeries(
            times=[2000 * u.years],
            values=[2218 * u.kg_CO2 / u.tonne_coal_bit],
            default_value=2344 * u.kg_CO2 / u.tonne_coal_bit,
            t_unit=u.years,
            )
        val_d = {}
        val_d[Geo.NL] = row_1_2
        val_d[Geo.NS] = row_1_2
        val_d[Geo.PE] = row_1_2
        val_d[Geo.QC] = row_1_2

        row_3_4 = SparseTimeSeries(
            times=[2010 * u.years],
            values=[2212 * u.kg_CO2 / u.tonne_coal_bit],
            default_value=2333 * u.kg_CO2 / u.tonne_coal_bit,
            t_unit=u.years,
            )
        val_d[Geo.NB] = row_3_4

        val_d[Geo.ON] = 2212 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.MB] = 2212 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.SK] = 2212 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.AB] = 2212 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.BC] = 2212 * u.kg_CO2 / u.tonne_coal_bit
        super().__init__(val_d=val_d)

class A6_1_10_ImportedBituminous(PTValues):

    def __init__(self):

        val_d = {}
        val_d[Geo.NB] = 2571 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.NS] = 2571 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.PE] = 2571 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.NL] = 2571 * u.kg_CO2 / u.tonne_coal_bit

        val_d[Geo.MB] = 2651 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.ON] = 2651 * u.kg_CO2 / u.tonne_coal_bit

        val_d[Geo.QC] = 2662 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.AB] = 2662 * u.kg_CO2 / u.tonne_coal_bit
        val_d[Geo.BC] = 2662 * u.kg_CO2 / u.tonne_coal_bit
        super().__init__(val_d=val_d)


class A6_1_10_Lignite(PTValues):
    def __init__(self):
        val_d = {}
        for pt in Geo.provinces_and_territories():
            val_d[pt] = 1463 * u.kg_CO2 / u.tonne_lignite
        super().__init__(val_d=val_d)


class A6_1_10_Subbituminous(PTValues):

    def __init__(self):
        NB = SparseTimeSeries(
            times=[
                2010 * u.years,
                2011 * u.years,
                2012 * u.years,
                2013 * u.years,
                2014 * u.years,
                2015 * u.years],
            values=[
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2352 * u.kg_CO2 / u.tonne_coal_subbit,
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2352 * u.kg_CO2 / u.tonne_coal_subbit],
            default_value=None,
            t_unit=u.years,
            )

        val_d = {}
        val_d[Geo.QC] = 1865 * u.kg_CO2 / u.tonne_coal_subbit
        val_d[Geo.ON] = 1865 * u.kg_CO2 / u.tonne_coal_subbit
        val_d[Geo.MB] = 1865 * u.kg_CO2 / u.tonne_coal_subbit

        val_d[Geo.NS] = 1743 * u.kg_CO2 / u.tonne_coal_subbit
        val_d[Geo.PE] = 1743 * u.kg_CO2 / u.tonne_coal_subbit

        val_d[Geo.SK] = 1775 * u.kg_CO2 / u.tonne_coal_subbit
        val_d[Geo.AB] = 1775 * u.kg_CO2 / u.tonne_coal_subbit
        val_d[Geo.BC] = 1775 * u.kg_CO2 / u.tonne_coal_subbit

        val_d[Geo.NB] = NB
        super().__init__(val_d=val_d)


class A6_1_10(object):
    # Construct the PTValues of coefficients by CoalType

    @staticmethod
    def ptv_by_coal_type():
        rval = {}
        rval[CoalType.CanadianBituminous] = A6_1_10_CanadianBituminous()
        rval[CoalType.ImportedBituminous] = A6_1_10_ImportedBituminous()
        rval[CoalType.Lignite] = A6_1_10_Lignite()
        rval[CoalType.CanadianSubbituminous] = A6_1_10_Subbituminous()
        rval[CoalType.ImportedSubbituminous] = A6_1_10_Subbituminous()
        return rval


class AmountConsumed(PTValues):
    coal_type: object

    def __init__(self, coal_type):
        # Good 2005-2021 inclusive
        scp = Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()
        ptv = scp.ptv_by_coal_type(coal_type)
        super().__init__(val_d=ptv.val_d, coal_type=coal_type)

    def scatter(self):
        super().scatter()
        plt.title(f'Coal for Electricity ({self.coal_type.value})')
        plt.legend(loc='upper right')

    @classmethod
    def all_ptvs(cls):
        rval = {coal_type: cls(coal_type) for coal_type in CoalType}
        for coal_type, ptv in rval.items():
            ptv.replace_CA_with_PX()
        return rval


class A6_1_10_Anthracite(PTValues):

    def __init__(self):
        val_d = {}
        for pt in Geo.provinces_and_territories():
            val_d[pt] = 3097 * u.kg_CO2 / u.tonne
        super().__init__(val_d=val_d)


class A6_1_12(object):

    def __init__(self):
        coef = .02
        self.CH4_by_coal_type = {
            CoalType.CanadianBituminous: coef * u.g_CH4 / u.kg_coal_bit,
            CoalType.ImportedBituminous: coef * u.g_CH4 / u.kg_coal_bit,
            CoalType.CanadianSubbituminous: coef * u.g_CH4 / u.kg_coal_subbit,
            CoalType.ImportedSubbituminous: coef * u.g_CH4 / u.kg_coal_subbit,
            CoalType.Lignite: coef * u.g_CH4 / u.kg_lignite,
        }

        coef = .03
        self.N2O_by_coal_type = {
            CoalType.CanadianBituminous: coef * u.g_N2O / u.kg_coal_bit,
            CoalType.ImportedBituminous: coef * u.g_N2O / u.kg_coal_bit,
            CoalType.CanadianSubbituminous: coef * u.g_N2O / u.kg_coal_subbit,
            CoalType.ImportedSubbituminous: coef * u.g_N2O / u.kg_coal_subbit,
            CoalType.Lignite: coef * u.g_N2O / u.kg_lignite,
        }



class CO2(PTValues):
    def __init__(self):
        amounts = AmountConsumed.all_ptvs()
        coefs = A6_1_10.ptv_by_coal_type()
        total = PTValues(val_d={})
        for coal_type, ptv in amounts.items():
            total += amounts[coal_type] * coefs[coal_type]
        super().__init__(val_d=total.val_d)


class CH4(PTValues):
    def __init__(self):
        amounts = AmountConsumed.all_ptvs()
        coefs = A6_1_12()
        total = PTValues(val_d={})
        for coal_type, ptv in amounts.items():
            total += amounts[coal_type] * coefs.CH4_by_coal_type[coal_type]
        super().__init__(val_d=total.val_d)


class N2O(PTValues):
    def __init__(self):
        amounts = AmountConsumed.all_ptvs()
        coefs = A6_1_12()
        total = PTValues(val_d={})
        for coal_type, ptv in amounts.items():
            total += amounts[coal_type] * coefs.N2O_by_coal_type[coal_type]
        super().__init__(val_d=total.val_d)


def plot_delta_coal_for_electricity_generation():
    co2 = CO2()
    ch4 = CH4()
    n2o = N2O()
    co2e = CO2e_from_emissions(co2, ch4, n2o)

    plt.figure()

    est = co2e.national_total()
    est.plot(label='Estimate')
    a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['coal']
    a13_ng.to(co2e.v_unit).plot(label='Annex13 (Target)')
    plt.title('Emissions: Electricity from Coal')
    plt.legend(loc='upper right')
    plt.xlim(2004, max(max(est.times), max(a13_ng.times)) + 1)
