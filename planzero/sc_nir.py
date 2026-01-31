from enum import Enum
import functools

from stats_can.sc import (pathlib, pd, zipfile, parse_tables, download_tables)
import stats_can
import markdown
import pandas as pd

from .ureg import u, Geo
from .ureg import ElectricityGenerationTech
from .sts import annual_report


def sc_metadata_by_product_id(productId):
    return stats_can.schemas.CubeMetadata(**stats_can.scwds.get_cube_metadata(productId)[0])


def markdown_from_dimension(scdim):
    lines = []
    # print(scdim)
    lines.append(f'## Dimension {scdim["dimensionPositionId"]}: {scdim["dimensionNameEn"]}')
    member_by_id = {member['memberId']: member for member in scdim['member']}
    children_by_id = {member['memberId']: [] for member in scdim['member']}
    printed_by_id = {member['memberId']: False for member in scdim['member']}
    indent_by_id = {member['memberId']: 0 for member in scdim['member']}

    for mid, member in member_by_id.items():
        if member['parentMemberId'] is not None:
            children_by_id[member['parentMemberId']].append(mid)
            indent_by_id[mid] = indent_by_id[member['parentMemberId']] + 2
    def append_children(_mid):
        for cid in children_by_id[_mid]:
            if not printed_by_id[cid]:
                lines.append(' ' * indent_by_id[cid] + '* ' + member_by_id[cid]['memberNameEn'])
                printed_by_id[cid] = True
                append_children(cid)

    for mid, member in member_by_id.items():
        if not printed_by_id[mid]:
            lines.append(' ' * indent_by_id[mid] + '* ' + member_by_id[mid]['memberNameEn'])
            printed_by_id[mid] = True
            append_children(mid)
    return '\n'.join(lines)


def jupyter_display_metadata(metadata):
    # import inside function so module can be imported without IPython installed
    from IPython.display import display, HTML
    md_text = (
        f'#{metadata["cubeTitleEn"]}'
        f' ([Stats-Can: {metadata["productId"]}](https://https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid={metadata["productId"]}))\n'
        + '\n\n'.join(markdown_from_dimension(dim) for dim in metadata['dimension']))
    return display(HTML(markdown.markdown(md_text)))


def zip_table_to_dataframe(
    table: str, path: pathlib.Path | None = None
) -> pd.DataFrame:
    """Read a StatsCan table into a pandas DataFrame.

    If a zip file of the table does not exist in path, downloads it

    Parameters
    ----------
    table
        the table to load to dataframe from zipped csv
    path
        where to download the tables or load them, default will go to current working directory

    Returns
    -------
    :
        the table as a dataframe
    """
    path = pathlib.Path(path) if path else pathlib.Path()
    # Parse tables returns a list, can only do one table at a time here though
    table = parse_tables(table)[0]
    table_zip = table + "-eng.zip"
    table_zip = path / table_zip
    if not table_zip.is_file():
        download_tables([table], path)
    csv_file = table + ".csv"
    with zipfile.ZipFile(table_zip) as myzip:
        with myzip.open(csv_file) as myfile:
            col_names = pd.read_csv(myfile, nrows=0).columns
        # reopen the file or it misses the first row
        with myzip.open(csv_file) as myfile:
            types_dict = {"VALUE": float}
            types_dict.update({col: str for col in col_names if col not in types_dict})
            df = pd.read_csv(myfile, dtype=types_dict)

    possible_cats = [
        "GEO",
        "DGUID",
        "STATUS",
        "SYMBOL",
        "TERMINATED",
        "DECIMALS",
        "UOM",
        "UOM_ID",
        "SCALAR_FACTOR",
        "SCALAR_ID",
        "VECTOR",
        "COORDINATE",
        "Wages",
        "National Occupational Classification for Statistics (NOC-S)",
        "Supplementary unemployment rates",
        "Sex",
        "Age group",
        "Labour force characteristics",
        "Statistics",
        "Data type",
        "Job permanency",
        "Union coverage",
        "Educational attainment",
    ]
    actual_cats = [col for col in possible_cats if col in col_names]
    df[actual_cats] = df[actual_cats].astype("category")
    # df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], errors="ignore") # JB: leads to assertion error so commenting it out
    return df


class StatsCanProduct(object):

    @classmethod
    @functools.cache
    def fetch_metadata(cls):
        return stats_can.scwds.get_cube_metadata(cls.product_id)[0]

    @classmethod
    @functools.cache
    def zip_table_to_dataframe(cls):
        rval = zip_table_to_dataframe(cls.product_id)
        rval['REF_DATE'] = rval['REF_DATE'].apply(pd.to_numeric)
        assert 'int' in str(rval.REF_DATE.values.dtype)
        return rval


class Supply_and_Demand_of_Natural_Gas_Liquids(StatsCanProduct):
    product_id = "25-10-0026-01"


class Supply_and_Demand_of_Primary_and_Secondary_Energy_TJ(StatsCanProduct):
    product_id = "25-10-0029-01"


class Supply_and_Demand_of_Primary_and_Secondary_Energy_NU(StatsCanProduct):
    product_id = "25-10-0030-01"


class Electric_Power_Annual_Generation_by_Class_of_Producer(StatsCanProduct):
    product_id = "25-10-0020-01"

    @classmethod
    @functools.cache
    def build_sts(cls, class_of_electricity_producer):
        """Return dict[ElectricityGenerationTech], dict[Geo, STS]]
        """
        df = cls.zip_table_to_dataframe()
        df = df[df['Class of electricity producer'] == class_of_electricity_producer]
        rval = {}
        for (geo_name, gen_type), geo_df in df.groupby(['GEO', 'Type of electricity generation']):
            if gen_type.startswith('Total'):
                continue
            geo = Geo(geo_name)
            gen_tech = ElectricityGenerationTech(gen_type)
            uom, = list(set(geo_df.UOM.values))
            rval.setdefault(gen_tech, {})
            rval[gen_tech][geo] = annual_report(
                times=geo_df.REF_DATE.values * u.years,
                values=geo_df.VALUE.values * u.Quantity(f'1 {uom.lower()}'),
                skip_nan_values=True)
        return rval

    @classmethod
    def utility_gen_by_tech_geo(cls):
        return cls.build_sts('Electricity producer, electric utilities')

    @classmethod
    def industry_gen_by_tech_geo(cls):
        return cls.build_sts('Electricity producer, industries')


conversion_factors_CO2 = {
    'Aviation gasoline': 69.0 * u.tonne / u.TJ,
    'Aviation turbo fuel': 71.5 * u.tonne / u.TJ,
    'Coke': 102.0 * u.tonnes / u.TJ,
    'Coke oven gas': 47.6 * u.tonnes / u.TJ, # Byproduct of steelmaking. High hydrogen content makes the CO₂ factor lower than coal
    'Diesel fuel oil': 69.9 * u.tonnes / u.TJ, # for trucks
    "Gas plant natural gas liquids (NGL's)": 63.0 * u.tonnes / u.TJ, # XXX this should be broken out into Ethane, Propane, Butane, and Pentanes Plus
    'Heavy fuel oil': 76.0 * u.tonnes / u.TJ, # marine diesel?
    'Kerosene and stove oil': 67.7 * u.tonnes / u.TJ,
    'Light fuel oil': 70.8 * u.tonnes / u.TJ, # aka furnace oil
    'Motor gasoline': 69.0 * u.tonne / u.TJ,
    'Natural gas': 50.0 * u.tonnes / u.TJ,
    'Non-energy products': 73.0 * u.tonnes / u.TJ, # these are non-energy products used for an energy use. Gemini explains this is e.g. combustion of waste lubricants
    'Petroleum coke': 96.5 * u.tonnes / u.TJ,
    'Primary electricity, hydro and nuclear':  0 * u.tonnes / u.TJ, # emissions are associated with production of electricity, not consumption
    "Refinery liquefied petroleum gases (LPG's)": 60.3 * u.tonnes / u.TJ, # Mostly Propane/Butane mix. Using Propane as the proxy as composition is unknown
    'Still gas': 49.8 * u.tonnes / u.TJ, # Gemini says this is refinery fuel
    'Steam': 0 * u.tonnes / u.TJ, # emissions are associated with production of steam, not consumption
    'Total coal': 87.0 * u.tonnes / u.TJ, # Gemini is listing Coal(bitumenous) and Coal (sub-bitumenous) at 85.5 and 91.0 respectively
}

thermal_efficiency_by_type_of_electricity_generation = {
    'Hydraulic turbine': None,
    'Tidal power turbine': None,
    'Wind power turbine': None,
    #'Conventional steam turbine': .34, # broken out by type of coal used in different provinces
    'Nuclear steam turbine': None,
    'Internal combustion turbine': .35, # diesel reciprocating engine generators
    'Combustion turbine': .30, # gas turbines of 2005-2014 era were mostly simple cycle
    'Other types of electricity generation': .25, # Gemini suggests burning wood chips or garbage
    'Geothermal': None,
}


def CO2_emissions_from_electricity_generation_2005():
    df = zip_table_to_dataframe("25-10-0020-01")
    # don't include electricity for industry, because it is counted as "producer consumption"
    # and already counted
    df = df[(df['REF_DATE'] == '2005')
            & (df['Class of electricity producer'] == 'Electricity producer, electric utilities') ]

    # natural gas
    ng_df = df[(df['GEO'] == 'Canada')
               & (df['Type of electricity generation'] == 'Combustion turbine')]

    ng_value, = ng_df['VALUE']
    ng_uom, = ng_df['UOM']
    ng_value_w_unit = u.Quantity(f'{ng_value} {ng_uom.lower()}')
    ng_CO2_inc = (
        ng_value_w_unit
        * conversion_factors_CO2['Natural gas']
        / thermal_efficiency_by_type_of_electricity_generation['Combustion turbine'])

    # diesel
    ic_df = df[(df['GEO'] == 'Canada')
               & (df['Type of electricity generation'] == 'Internal combustion turbine')]

    ic_value, = ic_df['VALUE']
    ic_uom, = ic_df['UOM']
    ic_value_w_unit = u.Quantity(f'{ic_value} {ic_uom.lower()}')
    ic_CO2_inc = (
        ic_value_w_unit
        * conversion_factors_CO2['Diesel fuel oil']
        / thermal_efficiency_by_type_of_electricity_generation['Internal combustion turbine'])

    # other
    ot_df = df[(df['GEO'] == 'Canada')
               & (df['Type of electricity generation'] == 'Other types of electricity generation')]

    if len(ot_df):
        ot_value, = ot_df['VALUE']
        ot_uom, = ot_df['UOM']
        ot_value_w_unit = u.Quantity(f'{ot_value} {ot_uom.lower()}')
        ot_CO2_inc = (
            ot_value_w_unit
            * conversion_factors_CO2['Diesel fuel oil']
            / thermal_efficiency_by_type_of_electricity_generation['Other types of electricity generation'])
    else:
        ot_CO2_inc = 0 * u.kg

    # coal
    co_df = df[(df['GEO'] != 'Canada')
               & (df['Type of electricity generation'] == 'Conventional steam turbine')]
    co_CO2_inc = 0 * u.kg
    for province, energy_mag, uom in zip(co_df['GEO'], co_df['VALUE'], co_df['UOM']):
        energy = u.Quantity(f'{energy_mag} {uom.lower()}')
        if province == 'Alberta':
            thermal_efficiency = .34 # sub-bit
        elif province == 'Saskatchewan':
            thermal_efficiency = .33 # lignite
        elif province in ['Newfoundland and Labrador', 'New Brunswick']:
            thermal_efficiency = .36 # bitumenous
        else:
            thermal_efficiency = .34 # dunno what it is
        co_CO2_inc += (
            energy * conversion_factors_CO2['Total coal'] / thermal_efficiency)

    return ng_CO2_inc + ic_CO2_inc + ot_CO2_inc + co_CO2_inc


def CO2_emissions_from_other_sources_2005_energy():
    df = Supply_and_Demand_of_Primary_and_Secondary_Energy_TJ.zip_table_to_dataframe()
    df = df[df['VALUE'].notna()
            & (df['GEO'] == 'Canada')
            & (df['REF_DATE'] == '2005')
            & (df['Supply and demand characteristics'].isin([
                'Producer consumption',
                'Energy use, final demand',
                'Transformed to steam generation',
                #'Transformed to electricity by utilities', -> see CO2_emissions_from_electricity_generation_2005
                #'Transformed to electricity by industry', -> see CO2_emissions_from_electricity_generation_2005
                #'Statistical difference',
                #'Other adjustments',
                ]))
           ]

    # The summation is over "Supply and demand characteristics
    combustion_by_fuel_type = {
        fuel_type: float(ft_df['VALUE'].sum()) * u.TJ
        for fuel_type, ft_df in df.groupby('Fuel type')}

    total_CO2 = 0 * u.tonnes
    for fuel_type, energy  in combustion_by_fuel_type.items():
        if fuel_type in ['Total primary and secondary energy', 'Total refined petroleum products', 'Primary energy']:
            # these are totals, subtotals, which would result in double-counting
            continue
        elif fuel_type in conversion_factors_CO2:
            # print('  ', fuel_type, consumed_TJ, conversion_factors_CO2[fuel_type] * consumed_TJ)
            total_CO2 += conversion_factors_CO2[fuel_type] * energy
        else:
            raise NotImplementedError(fuel_type)

    # for 2005
    total_CO2 += 15 * u.megatonnes # due to venting & flaring of natural gas
    # in general - there's a table for how much gas is flared
    # Gemini says Table 25-10-0085-01 but is sometimes wrong about table ids
    return total_CO2

# US emission coefficients 
# https://www.eia.gov/environment/emissions/co2_vol_mass.php

# Canadian emissions coefficients
# https://www.canada.ca/en/environment-climate-change/services/climate-change/pricing-pollution-how-it-will-work/output-based-pricing-system/federal-greenhouse-gas-offset-system/emission-factors-reference-values.html
# https://data-donnees.az.ec.gc.ca/api/file?path=%2Fsubstances%2Fmonitor%2Fcanada-s-official-greenhouse-gas-inventory%2FD-Emission-Factors%2FEN_Annex6_Emission_Factors.pdf
# -> table A6.1-1 lists CO2 emissions factors for marketable and non-marketable natural gas by province and by year
# -> table A6.1-7 is emissions for Petroleum Coke and Still gas over time

CO2_emission_coefficients_by_fuel = {
    'jet fuel': 21.50 * u.pounds / u.gallon,
    'aviation gas': 18.32 * u.pounds / u.gallon,
    'diesel': 2_681 * u.grams / u.liter,
    'coke oven gas': 687 * u.grams / (u.meter ** 3),
    'coke': 3173 * u.grams / u.kg,
    'heavy fuel oil': 3156 * u.grams / u.liter, # producer-consumption should be a bit higher
    'light fuel oil': 2753 * u.grams / u.liter, # producer-consumption should be a bit higher
    'kerosene': 2560 * u.grams / u.liter,
    'propane': 1515 * u.grams / u.liter,
    'ethane': 986 * u.grams / u.liter,
    'butane': 1747 * u.grams / u.liter,
    'gasoline': 2307 * u.grams / u.liter,
    'natural gas (marketable)': 1950 * u.grams / (u.meter ** 3), # TODO: by year, by province (A6.1-1)
    'natural gas (non-marketable)': 2200 * u.grams / (u.meter ** 3), # TODO: by year, by province (A6.1-2)
    'non-energy products': 1000 * u.grams / (u.meter ** 3), # TODO: no idea here
    'petroleum coke': 2800 * u.grams / u.liter, # TODO: by year, by refinery vs. upgrading facility
    'still gas': 1900 * u.grams / u.liter, # TODO: by year, by refinery vs. upgrading facility
    'lpg': 1.5 * u.kg / u.liter,  # TODO: ???
    'average coal': 2300 * u.kg / u.tonne, # TODO: there are so many kinds of coal
}




def CO2_emissions_from_other_sources_2005_natural_units():
    df = Supply_and_Demand_of_Primary_and_Secondary_Energy_NU.zip_table_to_dataframe()
    df = df[df['VALUE'].notna()
            & (df['GEO'] == 'Canada')
            & (df['REF_DATE'] == '2005')
            & (df['Supply and demand characteristics'].isin([
                'Producer consumption',
                'Energy use, final demand',
                'Transformed to steam generation',
                #'Transformed to electricity by utilities', -> see CO2_emissions_from_electricity_generation_2005
                #'Transformed to electricity by industry', -> see CO2_emissions_from_electricity_generation_2005
                #'Statistical difference',
                #'Other adjustments',
                ]))
           ]

    # The summation is over "Supply and demand characteristics
    combustion_by_fuel_type = {
        fuel_type: (float(ft_df['VALUE'].sum()), set(ft_df['UOM']))
        for fuel_type, ft_df in df.groupby('Fuel type')}

    total_CO2 = 0 * u.tonnes
    for fuel_type, (amount_mag, unit_strs)  in combustion_by_fuel_type.items():
        unit_str, = list(unit_strs)
        amount = u.Quantity(f'{amount_mag} {unit_str.lower()}')
        if fuel_type in ['Total refined petroleum products, secondary energy']:
            # these are totals, subtotals, which would result in double-counting
            continue
        elif fuel_type == 'Aviation gasoline, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['aviation gas']
        elif fuel_type == 'Aviation turbo fuel, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['jet fuel']
        elif fuel_type == 'Coke oven gas, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['coke oven gas']
        elif fuel_type == 'Coke, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['coke']
        elif fuel_type == 'Diesel fuel oil, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['diesel']
        elif fuel_type == "Gas plant natural gas liquids (NGL's), primary energy":
            total_CO2 += amount * (
                1/3 * CO2_emission_coefficients_by_fuel['propane']
                + 1/3 * CO2_emission_coefficients_by_fuel['ethane']
                + 1/3 * CO2_emission_coefficients_by_fuel['butane'])
        elif fuel_type == 'Heavy fuel oil, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['heavy fuel oil']
        elif fuel_type == 'Kerosene and stove oil, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['kerosene']
        elif fuel_type == 'Light fuel oil, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['light fuel oil']
        elif fuel_type == 'Motor gasoline, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['gasoline']
        elif fuel_type == 'Natural gas, primary energy':
            total_CO2 += amount * (
                .5 * CO2_emission_coefficients_by_fuel['natural gas (marketable)']
                + .5 * CO2_emission_coefficients_by_fuel['natural gas (non-marketable)'])
        elif fuel_type == 'Non-energy products, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['non-energy products']
        elif fuel_type == 'Petroleum coke, secondary energy':
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['petroleum coke']
        elif fuel_type == 'Primary electricity, hydro and nuclear, primary energy':
            pass
        elif fuel_type == "Refinery liquefied petroleum gases (LPG's), secondary energy":
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['lpg']
        elif fuel_type == "Steam, primary energy":
            pass
        elif fuel_type == "Still gas, secondary energy":
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['still gas']
        elif fuel_type == "Total coal, primary energy":
            total_CO2 += amount * CO2_emission_coefficients_by_fuel['average coal']
        else:
            print('  ', repr(fuel_type), amount)
            #raise NotImplementedError(fuel_type)

    # for 2005
    total_CO2 += 15 * u.megatonnes # due to venting & flaring of natural gas
    # in general - there's a table for how much gas is flared
    # Gemini says Table 25-10-0085-01 but is sometimes wrong about table ids
    return total_CO2
