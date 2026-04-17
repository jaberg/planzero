# https://www.petrinex.ca/PD/Documents/PD_Conventional_Volumetrics_Report.pdf

import enum
from .my_functools import cache
import json
import os

import pandas as pd
from pathlib import Path
import pint
import pydantic

from .ureg import u
from .enums import PT, GHG
from . import ghgrp
from . import objtensor
from . import sts
from . import pollution_waste_canada as pwc


class Month(str, enum.Enum):
    January = '01'
    February = '02'
    March = '03'
    April = '04'
    May = '05'
    June = '06'
    July = '07'
    August = '08'
    September = '09'
    October = '10'
    November = '11'
    December = '12'


class ActivityID(str, enum.Enum):
    # https://www.petrinex.gov.ab.ca/bbreportsSK/PRAActivityCodes.htm
    Difference = 'DIFF'
    Disposition = 'DISP'
    Emission = 'EMIS'
    Fire = 'FIRE'
    Flare = 'FLARE'
    FlaredOrWasted = 'FLARWAST'
    Fractionate = 'FRAC'
    Fuel = 'FUEL'
    FurtherProcessing = 'FURPROC'
    Imbalance = 'IMBAL'
    Injection = 'INJ'
    InventoryAdjustment = 'INVADJ'
    InventoryClose = 'INVCL'
    InventoryOpen = 'INVOP'
    LoadInjection = 'LDINJ'
    LoadInventoryAdjustment = 'LDINVADJ'
    LoadInventoryClose = 'LDINVCL'
    LoadInventoryOpen = 'LDINVOP'
    LoadRecovered = "LDREC"
    OilSandsMined = 'MINED'
    PlantUse = 'PLTUSE'
    ProcessToCreateProduct = 'PROC'
    Production = 'PROD'
    PurchaseDisposition = 'PURDISP'
    PurchaseReceipt = 'PURREC'
    Receipt = 'REC'
    Recycle = 'RECYC'
    Royalty = 'ROYALTY'
    Shrinkage = 'SHR'
    ShutIn = 'SHUTIN'
    Spillage = 'SPILL'

    StorageFacilityInjection = 'STINJ'
    StorageFacilityInventoryAdjustment = 'STINVADJ'
    StorageFacilityClosingInventory = 'STINVCL'
    StorageFacilityOpeningInventory = 'STINVOP'
    StorageFacilityRecovered = 'STREC'

    Theft = 'THEFT'
    Utilities = 'UTIL'
    Vent = 'VENT'


class ProductID(str, enum.Enum):
    # https://www.petrinex.gov.ab.ca/bbreportsSK/PRAProductCodes.htm
    AcidGas = 'ACGAS'
    Air = 'AIR'
    BrackishWater = 'BRKWTR'
    Brine = 'BRINE'
    MethaneMix = 'C1-MX'
    EthaneMix = 'C2-MX'
    EthaneSpec = 'C2-SP'
    PropaneMix = 'C3-MX'
    PropaneSpec = 'C3-SP'
    ButaneMix = 'C4-MX'
    ButaneSpec = 'C4-SP'
    PentanesMix = 'C5-MX'
    PentanesSpec = 'C5-SP'
    HexaneMix = 'C6-MX'
    HexaneSpec = 'C6-SP'
    CO2 = 'CO2'
    CO2Mix = 'CO2-MX'
    CO2Spec = 'CO2-SP'
    Condensate = 'COND'
    DieselOil = 'DIESEL'
    EntrainedGas = 'ENTGAS'
    FreshWater = 'FSHWTR'
    Gas = 'GAS' # Formation / Casing / Solution / Natural Gas
    Helium = 'HELIUM'
    IsoButaneMix = 'IC4-MX'
    IsoButaneSpec = 'IC4-SP'
    IsoPentaneMix = 'IC5-MX'
    IsoPentaneSpec = 'IC5-SP'
    LiteMix = 'LITEMX'
    Nitrogen = 'N2'
    NormalButaneMix = 'NC4-MX'
    NormalButaneSpec = 'NC4-SP'
    NormalPentaneMix = 'NC5-MX'
    NormalPentaneSpec = 'NC5-SP'
    Oxygen = 'O2'
    CrudeOil = 'OIL'
    Polymer = 'POLYMER'
    Sand = 'SAND'
    SulphurBasepad = 'SBASE'
    SulphurBlock = 'SBLOC'
    SulphurSprilled = 'SPRILL'
    Solvent = 'SOLV'
    Steam = 'STEAM'
    Sulphur = 'SUL'
    SyntheticCrude = 'SYNCRD'
    Waste = "WASTE"
    Water = "WATER"


UoM_by_ProdId = {
    ProductID.AcidGas: u.kilo_m3,
    ProductID.Air: u.kilo_m3,
    ProductID.CO2: u.kilo_m3,
    ProductID.EntrainedGas: u.kilo_m3,
    ProductID.Gas: u.kilo_m3,
    ProductID.Nitrogen: u.kilo_m3,
    ProductID.Oxygen: u.kilo_m3,
    ProductID.Solvent: u.kilo_m3,
}


class FacilityType(str, enum.Enum):
    Battery = 'BT'
    CustomTreating = 'CT'
    GasPlant = 'GP'
    GasGatheringSystem = 'GS'
    InjectionFacility = 'IF'
    Pipeline = 'PL'
    TankTerminal = 'TM'
    FreshFormationWaterSource = 'WT'


def read_volume_csv(year, month, pt):
    possible_paths = [
        f'data/petrinex/Vol_{year}-{month.value}-{pt.two_letter_code()}.csv.zip',
        f'data/petrinex/Vol_{year}-{month.value}-{pt.two_letter_code()}.CSV',
    ]
    for path in possible_paths:
        try:
            return pd.read_csv(
                path,
                converters = {
                    'Hours': lambda x: float('nan') if x == '***' or x == '' else float(x),
                })
        except IOError:
            continue
    raise IOError('no path match')


@cache
def ghgrp_id_by_petrinex_id(report_year=2022):
    df = ghgrp._read_emissions_sources() # 2022 & 2023
    npri_ids = df[ghgrp.ESKey.NPRI_ID]
    rval = {}
    n_skips = 0
    for npri_id in sorted(npri_ids.unique()):
        import requests
        if npri_id > 0:
            try:
                details = pwc.report_details(
                    int(npri_id),
                    report_year,
                    read_cache=True,
                    fetch=False,
                    write_cache=False)
            except pwc.NoDetailsAvailable as err:
                # TODO: log level
                #print('No details available', npri_id, err)
                n_skips += 1
                continue
            except requests.HTTPError as err:
                # TODO: log level
                #print('HTTP Error', npri_id, err)
                n_skips += 1
                continue
            gids = details.facility.ghgrp_ids()
            if gids:
                gid, = gids
                for pid in details.facility.petrinex_ids():
                    rval[pid] = gid
    if n_skips:
        print(f'ghgrp_id_by_petrinex_id skipped {n_skips} NPRI facilities mentioned in GHGRP')
    return rval


cache_dir = 'cache/petrinex'


def cache_path(year, month, pt, include_ghgrp, large_emitter_cutoff):
    if include_ghgrp:
        if large_emitter_cutoff is None:
            rval = Path(cache_dir) / f"volume-summary_{year}-{month.value}-{pt.two_letter_code()}.json"
        else:
            rval = Path(cache_dir) / f"volume-summary_cutoff-{large_emitter_cutoff}_{year}-{month.value}-{pt.two_letter_code()}.json"
    else:
        if large_emitter_cutoff is None:
            rval = Path(cache_dir) / f"volume-summary-noghgrp_{year}-{month.value}-{pt.two_letter_code()}.json"
        else:
            rval = Path(cache_dir) / f"volume-summary-noghgrp_cutoff-{large_emitter_cutoff}_{year}-{month.value}-{pt.two_letter_code()}.json"
    return rval


class VolumeSummary(pydantic.BaseModel):
    year: int
    month: Month
    pt: PT
    includes_ghgrp: bool
    volume_by_activity: dict[ActivityID, dict[ProductID, dict[FacilityType, float]]]
    large_emitter_cutoff: float | None = None

    @classmethod
    def new_from_df(cls, year, month, pt, df, gid_by_pid=None, large_emitter_cutoff=None):
        gid_by_pid = gid_by_pid or {}
        volume_by_activity = {}
        for aid, adf in df.groupby('ActivityID'):
            if aid != aid: # nan
                continue
            aid = ActivityID(aid)
            volume_by_activity.setdefault(aid, {})
            for prod_id, pdf in adf.groupby('ProductID'):
                if prod_id != prod_id: # nan
                    continue
                prod_id = ProductID(prod_id)
                volume_by_activity[aid].setdefault(prod_id, {})
                for ft, ftdf in pdf.groupby('ReportingFacilityType'):
                    ft = FacilityType(ft)
                    volume_by_activity[aid][prod_id].setdefault(ft, 0)
                    for row in ftdf.iloc:
                        if row.ReportingFacilityID in gid_by_pid:
                            continue
                        try:
                            assert row.Volume == row.Volume
                            volume_by_activity[aid][prod_id][ft] += float(row.Volume)
                        except (AssertionError, ValueError) as exc:
                            # volume could be e.g. "***"
                            print('Warning, skipping volume', fdf)
                            continue
        return cls(
            year=year,
            month=month,
            pt=pt,
            includes_ghgrp=not bool(gid_by_pid),
            volume_by_activity=volume_by_activity,
            large_emitter_cutoff=large_emitter_cutoff,
            )

    @property
    def cache_path(self):
        return cache_path(
            self.year, self.month, self.pt,
            include_ghgrp=self.includes_ghgrp,
            large_emitter_cutoff=self.large_emitter_cutoff,
            )

    def cache(self):
        os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load_cached(cls, year, month, pt, include_ghgrp, large_emitter_cutoff=None):
        path = cache_path(year, month, pt,
                          include_ghgrp=include_ghgrp,
                          large_emitter_cutoff=large_emitter_cutoff)
        with open(path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return cls.model_validate(cached_data)


@cache
def petrinex_annual_summary(pt, include_ghgrp, large_emitter_cutoff=None):
    rval = objtensor.empty(ProductID, ActivityID, FacilityType)
    tmp = {}
    basis_years = [2022, 2023, 2024]
    for year in basis_years:
        for month in Month:
            vsum = VolumeSummary.load_cached(
                year, month, pt,
                include_ghgrp=include_ghgrp,
                large_emitter_cutoff=large_emitter_cutoff)
            for aid in vsum.volume_by_activity:
                for pid in vsum.volume_by_activity[aid]:
                    tmp.setdefault(aid, {}).setdefault(pid, {})
                    for ft, amt in vsum.volume_by_activity[aid][pid].items():
                        tmp[aid][pid].setdefault(ft, {})
                        tmp[aid][pid][ft].setdefault(year, 0 * UoM_by_ProdId.get(pid, u.m3))
                        tmp[aid][pid][ft][year] += amt * UoM_by_ProdId.get(pid, u.m3)
    for pid in ProductID:
        rval[pid] = 0 * UoM_by_ProdId.get(pid, u.m3)
    for aid in tmp:
        for pid in tmp[aid]:
            for ft in tmp[aid][pid]:
                for year in basis_years:
                    tmp[aid][pid][ft].setdefault(year, 0 * UoM_by_ProdId.get(pid, u.m3))
                years, values = zip(*sorted(tmp[aid][pid][ft].items()))
                v_unit = values[0].u
                rval[pid, aid, ft] = sts.annual_report2(
                    years=years,
                    values=[vv.to(v_unit).magnitude for vv in values],
                    v_unit=v_unit)
    return rval


def petrinex_SK(include_ghgrp=False):
    return petrinex_annual_summary(pt=PT.SK, include_ghgrp=include_ghgrp)


def est_large_emitters(df, thresh):
    # TODO: try to estimate *all* facility emissions, including flaring and
    # venting
    rval = {}
    gas_fuel_df = df[(df['ActivityID'] == 'FUEL') & (df['ProductID'] == 'GAS')]
    for fac_id, fdf in gas_fuel_df.groupby('ReportingFacilityID'):
        # gas can come from different sources,
        # although I'm basically crossing my fingers that the *only* thing we're
        # summing over here is the FromToID / FromToIDIdentifier
        total_burned_volume_1e3m3 = fdf.Volume.sum()
        approx_g_CO2 = total_burned_volume_1e3m3 * 1e3 * 2441
        approx_kt_CO2e = approx_g_CO2 * 1e-9 * u.kt_CO2e
        if approx_kt_CO2e > thresh:
            rval[fac_id] = 'Unknown GHGRP Identifier'
    return rval


def main_build_cache(args):
    # My implementation assumes that facilities are either always
    # GHGRP-registered or never GHGRP-registered, it undercounts if e.g.
    # a facility operated for a while and then registered with the GHGRP.
    print('building ghrp_id by petrinex_id')
    if args.include_ghgrp:
        gid_by_pid = {}
    else:
        gid_by_pid = ghgrp_id_by_petrinex_id()
    pt = PT(args.PT)
    for month in Month:
        print('building cache file for', args.year, month, pt,
              'include_ghgrp', args.include_ghgrp,
              'exclude large-emitters above', args.large_emitter_cutoff_monthly)
        df = read_volume_csv(args.year, month, pt)
        # *remove* large estimators from the total
        # because they're supposed to be in the GHGRP data
        # even if we don't have all the tables for matching up facility IDs
        if args.large_emitter_cutoff_monthly is not None:
            thresh = args.large_emitter_cutoff_monthly * u.kt_CO2e
            by_pid = est_large_emitters(df, thresh)
            by_pid.update(gid_by_pid)
            unknowns = 0
            for pid, gid in by_pid.items():
                if gid == 'Unknown GHGRP Identifier':
                    unknowns += 1
            print(f'Ignoring {unknowns} unknown large emitters above monthly cutoff of {args.large_emitter_cutoff_monthly} kt CO2e/mo')
        else:
            by_pid = gid_by_pid
        vsum = VolumeSummary.new_from_df(args.year, month, pt, df, by_pid, args.large_emitter_cutoff_monthly)
        vsum.cache()
