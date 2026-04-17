"""
The Greenhouse Gas Reporting Program (GHGRP) offers public download of
facility-level data from

https://data-donnees.az.ec.gc.ca/data/substances/monitor/greenhouse-gas-reporting-program-ghgrp-facility-greenhouse-gas-ghg-data/


Unit tests verify that the NAICS_emissions totals match
the NAICS_source_emissions totals. The NAICS_source_emissions data is a superset
of information for the years that it covers, currently 2022 and 2023.

"""
import enum

import numpy as np
import pandas as pd

from .ureg import u, kt_by_ghg
from .naics import NAICS6
from .enums import GHG, PT
from .ghgvalues import GWP_100
from . import objtensor
from . import sts
from .my_functools import cache

class EKey(str, enum.Enum):
    GHGRP_ID = "GHGRP ID No. / No d'identification du PDGES"
    Year = 'Reference Year / Année de référence'
    Name = "Facility Name / Nom de l'installation"
    Location = "Facility Location / Emplacement de l'installation"
    City = "Facility City or District or Municipality / Ville ou District ou Municipalité de l'installation"
    Province = "Facility Province or Territory / Province ou territoire de l'installation"
    Postal_Code = "Facility Postal Code / Code postal de l'installation"
    Latitude = 'Latitude'
    Longitude = 'Longitude'
    NPRI_ID = "Facility NPRI ID / Numéro d'identification de l'INRP" # National Pollutant Release Inventory is sister program
    NAICS_Code = "Facility NAICS Code / Code SCIAN de l'installation"
    NAICS_Code_Description_En = "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais"
    NAICS_Code_Description_Fr = "French Facility NAICS Code Description / Description du code SCIAN de l'installation en français"
    RCo_Legal_Name = 'Reporting Company Legal Name / Dénomination sociale de la société déclarante'
    RCo_Trade_Name = 'Reporting Company Trade Name / Nom commercial de la société déclarante'
    RCo_Business_Number = "Reporting Company CRA Business Number / Numéro d'entreprise ARC de la société déclarante"
    RCo_DUNS_Number = 'Reporting Company DUNS Number / Numéro DUNS de la société déclarante' # International id for business entities
    PubCon_Name = 'Public Contact Name / Nom du responsable des renseignements au public'
    PubCon_Position = 'Public Contact Position / Poste ou Titre du responsable des renseignements au public'
    PubCon_Telephone = 'Public Contact Telephone / Numéro de téléphone du responsable des renseignements au public'
    PubCon_Ext = 'Public Contact Extension / Poste téléphonique du responsable des renseignements au public'
    PubCon_Email = 'Public Contact Email / Adresse électronique du responsable des renseignements au public'
    PubCon_Address = 'Public Contact Mailing Address / Adresse postale du responsable des renseignements au public'
    PubCon_City = 'Public Contact City or District or Municipality / Ville ou District ou Municipalité du responsable des renseignements au public'
    PubCon_Province = 'Public Contact Province or Territory / Province ou Territoire du responsable des renseignements au public'
    PubCon_Postal_Code = 'Public Contact Postal Code / Code postal du responsable des renseignement au public'
    CO2 = 'CO2 (tonnes)'
    CH4 = 'CH4 (tonnes)'
    CH4_tCO2e = 'CH4 (tonnes CO2e / tonnes éq. CO2)'
    N2O = 'N2O (tonnes)'
    N2O_tCO2e = 'N2O (tonnes CO2e / tonnes éq. CO2)'
    HFC_23 = 'HFC-23 (tonnes)'
    HFC_23_tCO2e = 'HFC-23 (tonnes CO2e / tonnes éq. CO2)'
    HFC_32 = 'HFC-32 (tonnes)'
    HFC_32_tCO2e = 'HFC-32 (tonnes CO2e / tonnes éq. CO2)'
    HFC_41 = 'HFC-41 (tonnes)'
    HFC_41_tCO2e = 'HFC-41 (tonnes CO2e / tonnes éq. CO2)'
    HFC_43 = 'HFC-43-10mee (tonnes)'
    HFC_43_tCO2e = 'HFC-43-10mee (tonnes CO2e / tonnes éq. CO2)'
    HFC_125 = 'HFC-125 (tonnes)'
    HFC_125_tCO2e = 'HFC-125 (tonnes CO2e / tonnes éq. CO2)'
    HFC_134 = 'HFC-134 (tonnes)'
    HFC_134_tCO2e = 'HFC-134 (tonnes CO2e / tonnes éq. CO2)'
    HFC_134a = 'HFC-134a (tonnes)'
    HFC_134a_tCO2e = 'HFC-134a (tonnes CO2e / tonnes éq. CO2)'
    HFC_143 = 'HFC-143 (tonnes)'
    HFC_143_tCO2e = 'HFC-143 (tonnes CO2e / tonnes éq. CO2)'
    HFC_143a = 'HFC-143a (tonnes)'
    HFC_143a_tCO2e = 'HFC-143a (tonnes CO2e / tonnes éq. CO2)'
    HFC_152_a = 'HFC-152a (tonnes)'
    HFC_152_a_tCO2e = 'HFC-152a (tonnes CO2e / tonnes éq. CO2)'
    HFC_227ea = 'HFC-227ea (tonnes)'
    HFC_227ea_tCO2e = 'HFC-227ea (tonnes CO2e / tonnes éq. CO2)'
    HFC_236fa = 'HFC-236fa (tonnes)'
    HFC_236fa_tCO2e = 'HFC-236fa (tonnes CO2e / tonnes éq. CO2)'
    HFC_245ca = 'HFC-245ca (tonnes)'
    HFC_245ca_tCO2e = 'HFC-245ca (tonnes CO2e / tonnes éq. CO2)'
    HFC_tCO2e = 'HFC Total (tonnes CO2e / tonnes éq. CO2)'
    CF4 = 'CF4 (tonnes)'
    CF4_tCO2e = 'CF4 (tonnes CO2e / tonnes éq. CO2)'
    C2F6 = 'C2F6 (tonnes)'
    C2F6_tCO2e = 'C2F6 (tonnes CO2e / tonnes éq. CO2)'
    C3F8 = 'C3F8 (tonnes)'
    C3F8_tCO2e = 'C3F8 (tonnes CO2e / tonnes éq. CO2)'
    C4F10 = 'C4F10 (tonnes)'
    C4F10_tCO2e = 'C4F10 (tonnes CO2e / tonnes éq. CO2)'
    C4F8 = 'C4F8 (tonnes)'
    C4F8_tCO2e = 'C4F8 (tonnes CO2e / tonnes éq. CO2)'
    C5F12 = 'C5F12 (tonnes)'
    C5F12_tCO2e = 'C5F12 (tonnes CO2e / tonnes éq. CO2)'
    C6F14 = 'C6F14 (tonnes)'
    C6F14_tCO2e = 'C6F14 (tonnes CO2e / tonnes éq. CO2)'
    PFC_tCO2e = 'PFC Total (tonnes CO2e / tonnes éq. CO2)'
    SF6 = 'SF6 (tonnes)'
    SF6_tCO2e = 'SF6 (tonnes CO2e / tonnes éq. CO2)'
    TotalEmissions_tCO2e = "Total Emissions (tonnes CO2e) / Émissions totales (tonnes éq. CO2)"
    GHGRP_Quant_Req = 'GHGRP Quantification Requirements / Exigences de quantification du PDGES'
    Emission_Factors = "Emission Factors / Coefficients d'émission"
    Engineering_Estimates = 'Engineering Estimates / Estimations techniques'
    Mass_Balance = 'Mass Balance / Bilan massique'
    Monitoring_or_Direct_Measurement = 'Monitoring or Direct Measurement / Surveillance ou mesure directe'

EKeyGHGs = {
    EKey.CO2: GHG.CO2,
    EKey.CH4: GHG.CH4,
    EKey.N2O: GHG.N2O,
    EKey.HFC_tCO2e: GHG.HFCs,
    EKey.PFC_tCO2e: GHG.PFCs,
    EKey.SF6: GHG.SF6}

class ESKey(str, enum.Enum):
    GHGRP_ID = "Facility GHGRP ID No. / Installation No d'identification du PDGES"
    Year = 'Reference Year / Année de référence'
    Name = "Facility Name / Nom de l'installation"
    Location = "Facility Location / Emplacement de l'installation"
    City = "Facility City or District or Municipality / Ville ou District ou Municipalité de l'installation"
    Province = "Facility Province or Territory / Province ou territoire de l'installation"
    Postal_Code = "Facility Postal Code / Code postal de l'installation"
    Latitude = 'Latitude'
    Longitude = 'Longitude'
    NPRI_ID = "Facility NPRI ID / Numéro d'identification de l'INRP" # National Pollutant Release Inventory is sister program
    NAICS_Code = "Facility NAICS Code / Code SCIAN de l'installation"
    NAICS_Code_Description_En = "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais"
    NAICS_Code_Description_Fr = "French Facility NAICS Code Description / Description du code SCIAN de l'installation en français"
    RCo_Legal_Name = 'Reporting Company Legal Name / Dénomination sociale de la société déclarante'
    RCo_Trade_Name = 'Reporting Company Trade Name / Nom commercial de la société déclarante'
    RCo_Business_Number = "Reporting Company CRA Business Number / Numéro d'entreprise ARC de la société déclarante"
    RCo_DUNS_Number = 'Reporting Company DUNS Number / Numéro DUNS de la société déclarante' # International id for business entities
    PubCon_Name = 'Public Contact Name / Nom du responsable des renseignements au public'
    PubCon_Position = 'Public Contact Position / Poste ou Titre du responsable des renseignements au public'
    PubCon_Telephone = 'Public Contact Telephone / Numéro de téléphone du responsable des renseignements au public'
    PubCon_Ext = 'Public Contact Extension / Poste téléphonique du responsable des renseignements au public'
    PubCon_Email = 'Public Contact Email / Adresse électronique du responsable des renseignements au public'
    PubCon_Address = 'Public Contact Mailing Address / Adresse postale du responsable des renseignements au public'
    PubCon_City = 'Public Contact City or District or Municipality / Ville ou District ou Municipalité du responsable des renseignements au public'
    PubCon_Province = 'Public Contact Province or Territory / Province ou Territoire du responsable des renseignements au public'
    PubCon_Postal_Code = 'Public Contact Postal Code / Code postal du responsable des renseignement au public'
    Emission_Source = "Emission Source / Source d'emission"
    CO2 = 'CO2 (tonnes)'
    CH4 = 'CH4 (tonnes)'
    CH4_tCO2e = 'CH4 (tonnes CO2e / tonnes éq. CO2)'
    N2O = 'N2O (tonnes)'
    N2O_tCO2e = 'N2O (tonnes CO2e / tonnes éq. CO2)'
    HFC_23 = 'HFC-23 (tonnes)'
    HFC_23_tCO2e = 'HFC-23 (tonnes CO2e / tonnes éq. CO2)'
    HFC_32 = 'HFC-32 (tonnes)'
    HFC_32_tCO2e = 'HFC-32 (tonnes CO2e / tonnes éq. CO2)'
    HFC_41 = 'HFC-41 (tonnes)'
    HFC_41_tCO2e = 'HFC-41 (tonnes CO2e / tonnes éq. CO2)'
    HFC_43 = 'HFC-43-10mee (tonnes)'
    HFC_43_tCO2e = 'HFC-43-10mee (tonnes CO2e / tonnes éq. CO2)'
    HFC_125 = 'HFC-125 (tonnes)'
    HFC_125_tCO2e = 'HFC-125 (tonnes CO2e / tonnes éq. CO2)'
    HFC_134 = 'HFC-134 (tonnes)'
    HFC_134_tCO2e = 'HFC-134 (tonnes CO2e / tonnes éq. CO2)'
    HFC_134a = 'HFC-134a (tonnes)'
    HFC_134a_tCO2e = 'HFC-134a (tonnes CO2e / tonnes éq. CO2)'
    HFC_143 = 'HFC-143 (tonnes)'
    HFC_143_tCO2e = 'HFC-143 (tonnes CO2e / tonnes éq. CO2)'
    HFC_143a = 'HFC-143a (tonnes)'
    HFC_143a_tCO2e = 'HFC-143a (tonnes CO2e / tonnes éq. CO2)'
    HFC_152_a = 'HFC-152a (tonnes)'
    HFC_152_a_tCO2e = 'HFC-152a (tonnes CO2e / tonnes éq. CO2)'
    HFC_227ea = 'HFC-227ea (tonnes)'
    HFC_227ea_tCO2e = 'HFC-227ea (tonnes CO2e / tonnes éq. CO2)'
    HFC_236fa = 'HFC-236fa (tonnes)'
    HFC_236fa_tCO2e = 'HFC-236fa (tonnes CO2e / tonnes éq. CO2)'
    HFC_245ca = 'HFC-245ca (tonnes)'
    HFC_245ca_tCO2e = 'HFC-245ca (tonnes CO2e / tonnes éq. CO2)'
    HFC_tCO2e = 'HFC Total (tonnes CO2e / tonnes éq. CO2)'
    CF4 = 'CF4 (tonnes)'
    CF4_tCO2e = 'CF4 (tonnes CO2e / tonnes éq. CO2)'
    C2F6 = 'C2F6 (tonnes)'
    C2F6_tCO2e = 'C2F6 (tonnes CO2e / tonnes éq. CO2)'
    C3F8 = 'C3F8 (tonnes)'
    C3F8_tCO2e = 'C3F8 (tonnes CO2e / tonnes éq. CO2)'
    C4F10 = 'C4F10 (tonnes)'
    C4F10_tCO2e = 'C4F10 (tonnes CO2e / tonnes éq. CO2)'
    C4F8 = 'C4F8 (tonnes)'
    C4F8_tCO2e = 'C4F8 (tonnes CO2e / tonnes éq. CO2)'
    C5F12 = 'C5F12 (tonnes)'
    C5F12_tCO2e = 'C5F12 (tonnes CO2e / tonnes éq. CO2)'
    C6F14 = 'C6F14 (tonnes)'
    C6F14_tCO2e = 'C6F14 (tonnes CO2e / tonnes éq. CO2)'
    PFC_tCO2e = 'PFC Total (tonnes CO2e / tonnes éq. CO2)'
    SF6 = 'SF6 (tonnes)'
    SF6_tCO2e = 'SF6 (tonnes CO2e / tonnes éq. CO2)'
    TotalEmissionsFromSource_tCO2e = 'Total Emissions from Source (tonnes CO2e) / Émissions totales de la Source (tonnes éq. CO2)'
    TotalEmissionsFromFacility_tCO2e = "Total Facility Emissions (tonnes CO2e) / Émissions totales de l'installation (tonnes éq. CO2)"
    GHGRP_Quant_Req = 'GHGRP Quantification Requirements / Exigences de quantification du PDGES'
    Emission_Factors = "Emission Factors / Coefficients d'émission"
    Engineering_Estimates = 'Engineering Estimates / Estimations techniques'
    Mass_Balance = 'Mass Balance / Bilan massique'
    Monitoring_or_Direct_Measurement = 'Monitoring or Direct Measurement / Surveillance ou mesure directe'

ESKeyGHGs = {
    ESKey.CO2: GHG.CO2,
    ESKey.CH4: GHG.CH4,
    ESKey.N2O: GHG.N2O,
    ESKey.HFC_tCO2e: GHG.HFCs,
    ESKey.PFC_tCO2e: GHG.PFCs,
    ESKey.SF6: GHG.SF6,
    # Where's NF3?
}
ESKeyGHGs_tCO2e = [ESKey.CO2, ESKey.CH4_tCO2e, ESKey.N2O_tCO2e, ESKey.HFC_tCO2e, ESKey.PFC_tCO2e, ESKey.SF6_tCO2e]

v_unit_by_EKey = {
    EKey.TotalEmissions_tCO2e: u.tonne_CO2e,
    EKey.CO2: u.tonne_CO2,
    EKey.CH4: u.tonne_CH4,
    EKey.N2O: u.tonne_N2O,
    EKey.HFC_tCO2e: u.tonne_CO2e / GWP_100[GHG.HFCs],
    EKey.PFC_tCO2e: u.tonne_CO2e / GWP_100[GHG.PFCs],
    EKey.SF6: u.tonne_SF6,
}

quant_by_ESKey = {
    ESKey.TotalEmissionsFromSource_tCO2e: 1 * u.tonne_CO2e,
    ESKey.TotalEmissionsFromFacility_tCO2e: 1 * u.tonne_CO2e,
    ESKey.CO2: 1 * u.tonne_CO2,
    ESKey.CH4: 1 * u.tonne_CH4,
    ESKey.N2O: 1 * u.tonne_N2O,
    ESKey.HFC_tCO2e: u.tonne_CO2e / GWP_100[GHG.HFCs],
    ESKey.PFC_tCO2e: u.tonne_CO2e / GWP_100[GHG.PFCs],
    ESKey.SF6: 1 * u.tonne_SF6,
}


class EmissionSource(str, enum.Enum):
    StationaryFuelCombustion = 'Stationary Fuel Combustion / Combustion stationnaire de combustibles'
    OnSiteTransport = 'On-site Transportation / Transport sur le site'
    Waste = 'Waste / Déchets'
    Leakage = 'Leakage / Fuites'
    Venting = 'Venting / Évacuation'
    Flaring = 'Flaring / Torchage'
    IndustrialProcess = 'Industrial Processes / Procédés industriels'
    Wastewater = 'Wastewater / Eaux usées'
    Unspecified = 'unspecified' # not every row in the source file indicates an emission source


@cache
def _read_emissions():
    df = pd.read_csv('data/PDGES-GHGRP-GHGEmissionsGES-2004-Present.csv')
    return df


@cache
def _read_emissions_sources():
    df = pd.read_csv('data/PDGES-GHGRP-GHGEmissionsSourcesGES-2022-2023.csv')
    return df


def GHGRP_IDs():
    df = _read_emissions()
    return list(sorted(df[EKey.GHGRP_ID].unique()))


@cache
def facilities_by_NAICS():
    """Return the set of facilities that has, at any time, reported under each NAICS code.

    N.B. There are instances of facilities reporting under different codes in different years.
    """
    rval = {}
    missing = False
    df = _read_emissions()
    for naics, ndf in _read_emissions().groupby(EKey.NAICS_Code):
        try:
            rval[NAICS6(naics)] = list(sorted(ndf[EKey.GHGRP_ID].unique()))
        except ValueError :
            desc, = ndf[EKey.NAICS_Code_Description_En].unique()
            print(desc.replace(' ', '_').replace('(', '_').replace(')', '').replace(',', ''), '=', naics)
            missing = True
    assert not missing
    return rval


def _NAICS_source_emission_dict(nan_value_as_zero):
    # (naics, emission source, eskey of ghg, pt) -> year -> (sum of facility values)
    rval = {}
    err = False
    for row in _read_emissions_sources().iloc:
        naics = NAICS6(row[ESKey.NAICS_Code])
        if row[ESKey.Province] != row[ESKey.Province]: # nan
            pt = PT.XX
        else:
            pt = PT(row[ESKey.Province])
        try:
            emission_source = EmissionSource(row[ESKey.Emission_Source])
        except ValueError:
            assert np.isnan(row[ESKey.Emission_Source])
            emission_source = EmissionSource.Unspecified

        year = int(row[ESKey.Year])

        eskey_co2e_sum = 0
        for es_ghg, eskey_tCO2e in zip(ESKeyGHGs, ESKeyGHGs_tCO2e):
            value = float(row[es_ghg])
            if nan_value_as_zero and value != value:
                value = 0

            value_tCO2e = float(row[eskey_tCO2e])
            if nan_value_as_zero and value_tCO2e != value_tCO2e:
                value_tCO2e = 0
            eskey_co2e_sum += value_tCO2e

            if value_tCO2e == 0:
                assert value == 0

            key = (naics, emission_source, es_ghg, pt)

            rval.setdefault(key, {}).setdefault(year, 0)
            rval[key][year] += value
        target = row[ESKey.TotalEmissionsFromSource_tCO2e]
        if not np.allclose(eskey_co2e_sum, target):
            assert emission_source == EmissionSource.Unspecified
            assert eskey_co2e_sum == 0
            # a facility has reported emissions without identifying which gases they were
            # or for what reason they were emitted.
            # Here, we need to associate the emissions with a gas.
            # For convenience, we'll pick CO2
            rval[naics, emission_source, ESKey.CO2, pt][year] += target
    assert not err
    return rval


def GHG_NAICS_source_emissions(
        nan_value_as_zero=False,
        source_emission_dict=None,
        min_year_inclusive=2022,
        max_year_exclusive=2024):
    """Return NAICS category subtotals by summing over facilities for each
    year.

    This is better for emissions accounting than facility_source_emissions in
    principle because a facility's NAICS code may change from year to year.
    """
    rval = objtensor.empty(GHG, NAICS6, EmissionSource, PT)
    for ghg in GHG:
        rval[ghg].fill(0 * kt_by_ghg[ghg])
    source_emission_dict = source_emission_dict or _NAICS_source_emission_dict(nan_value_as_zero)
    basis_years = list(range(min_year_inclusive, max_year_exclusive))
    for (naics, em_src, es_ghg, pt), val_by_year in source_emission_dict.items():
        val_by_year = dict(val_by_year) # copy
        for yy in basis_years:
            val_by_year.setdefault(yy, 0) # update the copy in-place
        years, values = zip(*sorted(val_by_year.items()))
        assert min_year_inclusive <= years[0]
        assert max_year_exclusive > years[-1]
        # normally the pattern is just to get unit by ghg, but
        # we need magnitude and unit in this case
        quant = quant_by_ESKey[es_ghg]
        rval[ESKeyGHGs[es_ghg], naics, em_src, pt] = sts.annual_report2(
            years=years,
            values=[vv * quant.magnitude for vv in values],
            v_unit=quant.u)
    return rval


@cache
def NAICS_emissions(ekey):
    """Return NAICS category subtotals by summing over facilities for each
    year.

    This is better for emissions accounting than facility_source_emissions, at
    least in principle, because a facility's NAICS code may change from year to year.
    """
    tmp = {} # naics -> year -> (sum of facility values)
    min_year_inclusive = 2004
    max_year_exclusive = 2024
    for row in _read_emissions().iloc:
        naics = NAICS6(row[EKey.NAICS_Code])

        year = int(row[EKey.Year])
        assert min_year_inclusive <= year < max_year_exclusive

        value = float(row[ekey])
        assert value == value # not nan

        tmp.setdefault(naics, {})
        tmp[naics].setdefault(year, 0)
        tmp[naics][year] += value
    rval = objtensor.empty(NAICS6)
    rval.fill(0 * v_unit_by_EKey[ekey])
    basis_years = list(range(min_year_inclusive, max_year_exclusive))
    for naics in tmp:
        dd = tmp[naics]
        for yy in basis_years:
            dd.setdefault(yy, 0)
        years, values = zip(*sorted(dd.items()))
        rval[naics] = sts.annual_report2(
            years=years,
            values=values,
            v_unit=v_unit_by_EKey[ekey])
    return rval


@cache
def source_emissions_backfill_proportions():
    rval = {} # facility_id -> emissions source -> proportion
    for row in _read_emissions_sources().iloc:
        year = int(row[ESKey.Year])
        if year == 2022:
            fid = row[ESKey.GHGRP_ID]
            try:
                emission_source = EmissionSource(row[ESKey.Emission_Source])
            except ValueError:
                assert np.isnan(row[ESKey.Emission_Source])
                emission_source = EmissionSource.Unspecified

            from_source = float(row[ESKey.TotalEmissionsFromSource_tCO2e])
            #from_facility = float(row[ESKey.TotalEmissionsFromFacility_tCO2e])

            if from_source != from_source:
                from_source = 0 # why are there nans?
            #if from_facility != from_facility:
                #from_facility = 1e-6 # why are there nans?

            #assert from_facility >= from_source, (from_facility, from_source, fid)
            rval.setdefault(fid, {})
            assert emission_source not in rval[fid]
            rval[fid][emission_source] = from_source

    for fid, ed in rval.items():
        total_from_facility = sum(ed.values())
        if total_from_facility <= 0:
            raise NotImplementedError('net-negative facility in 2022')
        for esrc in EmissionSource:
            ed[esrc] = ed.get(esrc, 0) / total_from_facility
    return rval


@cache
def GHG_NAICS_source_emissions_backfilled():
    """Return NAICS category subtotals by summing over facilities for each
    year. For years prior to 2022, estimate emissions per EmissionSource using
    the prior year's total and the 2022 proportions.

    For facilities that are no-longer operating in 2022, estimate 80%
    StationaryFuelCombustion and 20% Unspecified.
    """
    source_emission_dict = _NAICS_source_emission_dict(nan_value_as_zero=True)
    fid_props = source_emissions_backfill_proportions()
    default_props = {
        EmissionSource.StationaryFuelCombustion: .8,
        EmissionSource.Unspecified: .2}
    for row in _read_emissions().iloc:
        year = int(row[EKey.Year])
        if year >= 2022:
            continue

        fid = row[EKey.GHGRP_ID]
        naics = NAICS6(row[EKey.NAICS_Code])
        if row[EKey.Province] != row[EKey.Province]: # nan
            pt = PT.XX
        else:
            pt = PT(row[EKey.Province])
        for es_ghg in ESKeyGHGs:
            value = float(row[es_ghg])
            if value != value:
                # is there a more-appropriate way to handle nan here?
                value = 0
            props = fid_props.get(fid, default_props)
            for em_src, esprop in props.items():
                key = (naics, em_src, es_ghg, pt)
                source_emission_dict.setdefault(key, {})
                source_emission_dict[key].setdefault(year, 0)
                source_emission_dict[key][year] += value * esprop
    return GHG_NAICS_source_emissions(
        nan_value_as_zero=True,
        source_emission_dict=source_emission_dict,
        min_year_inclusive=2004,
        max_year_exclusive=2024)
