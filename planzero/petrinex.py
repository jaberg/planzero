# https://www.petrinex.ca/PD/Documents/PD_Conventional_Volumetrics_Report.pdf

import enum
import functools
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
    InjectionFacillity = 'IF'
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


@functools.cache
def ghgrp_id_by_petrinex_id(report_year=2022):
    df = ghgrp._read_emissions_sources() # 2022 & 2023
    npri_ids = df[ghgrp.ESKey.NPRI_ID]
    rval = {}
    for npri_id in sorted(npri_ids.unique()):
        if npri_id > 0:
            try:
                details = pwc.report_details(
                    int(npri_id),
                    report_year,
                    read_cache=True,
                    fetch=False,
                    write_cache=False)
            except pwc.NoDetailsAvailable as err:
                print(err)
                continue
            gids = details.facility.ghgrp_ids()
            if gids:
                gid, = gids
                for pid in details.facility.petrinex_ids():
                    rval[pid] = gid
    return rval


cache_dir = 'cache/petrinex'


def cache_path(year, month, pt, include_ghgrp):
    if include_ghgrp:
        rval = Path(cache_dir) / f"volume-summary_{year}-{month.value}-{pt.two_letter_code()}.json"
    else:
        rval = Path(cache_dir) / f"volume-summary-noghgrp_{year}-{month.value}-{pt.two_letter_code()}.json"
    return rval


class VolumeSummary(pydantic.BaseModel):
    year: int
    month: Month
    pt: PT
    includes_ghgrp: bool
    volume_by_activity: dict[ActivityID, dict[ProductID, dict[FacilityType, float]]]

    @classmethod
    def new_from_df(cls, year, month, pt, df, gid_by_pid=None):
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
            volume_by_activity=volume_by_activity)

    @property
    def cache_path(self):
        return cache_path(
            self.year, self.month, self.pt,
            include_ghgrp=self.includes_ghgrp)

    def cache(self):
        os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load_cached(cls, year, month, pt, include_ghgrp):
        path = cache_path(year, month, pt, include_ghgrp=include_ghgrp)
        with open(path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return cls.model_validate(cached_data)


@functools.cache
def petrinex_annual_summary(pt, include_ghgrp=False):
    rval = objtensor.empty(ProductID, ActivityID, FacilityType)
    tmp = {}
    basis_years = [2022, 2023, 2024]
    for year in basis_years:
        for month in Month:
            vsum = VolumeSummary.load_cached(
                year, month, pt, include_ghgrp=include_ghgrp)
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
        print('building cache file for', args.year, month, pt, 'include_ghgrp', args.include_ghgrp)
        df = read_volume_csv(args.year, month, pt)
        vsum = VolumeSummary.new_from_df(
            args.year, month, pt, df, gid_by_pid)
        vsum.cache()
