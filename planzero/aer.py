"""AER: Alberta Energy Regulator"""

import enum
from .my_functools import cache

import pandas as pd
from .ureg import u
from . import objtensor
from . import sts


class VentingType(str, enum.Enum):
    Compressor = 'COMPRESSOR'
    PneumaticInstrument = 'PNEUMATICINSTRUMENT'
    PneumaticPump = 'PNEAMATICPUMP'
    Fugitive = 'FUGITIVE'
    RoutineVent = 'ROUTINEVENT'


# https://www.aer.ca/data-and-performance-reports/statistical-reports/st60b
@cache
def st60b_2024_OneStop():
    rval = objtensor.empty(VentingType)
    onestop_2023 = pd.read_excel('data/ST60B_2024.xlsx',
                                 sheet_name="2023 OneStop", header=6)
    onestop_2022 = pd.read_excel('data/ST60B_2024.xlsx',
                                 sheet_name="2022 OneStop", header=6)
    onestop_2021 = pd.read_excel('data/ST60B_2024.xlsx',
                                 sheet_name="2021 Onestop", header=6)
    onestop_2020 = pd.read_excel('data/ST60B_2024.xlsx',
                                 sheet_name="2020 Onestop", header=6)

    for vt in VentingType:
        rval[vt] = sts.annual_report2(
            years=[2020, 2021, 2022, 2023],
            values=[
                onestop_2020[vt.value].sum(),
                onestop_2021[vt.value].sum(),
                onestop_2022[vt.value].sum(),
                onestop_2023[vt.value].sum(),
            ],
            v_unit=u.m3_NG_nmk)
    return rval


# Data extracted from Alberta Energy Regulator (AER) Report VPR6301, Page 6
# https://static.aer.ca/prd/documents/sts/st13/st13a_2024_summary.pdf
# Units: 1000m3 for Gas/Gas equivalent, m3 for Liquids, MTS for Sulphur

st13a_2024 = {
    "Production Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Receipts: Alberta Facilities": [
        247_112_893.0, 253_883_398.1, 262_195_329.0, 275_380_165.2, 273_704_965.1,
        266098287.1, 280602429.6, 295316965.6, 297030650.4, 310415033.1
    ],
    "Receipts: Non-Alberta Facilities": [
        411143.5, 257491.0, 198467.0, 172233.6, 219987.1,
        198624.2, 198400.3, 198507.0, 153251.1, 198277.9
    ],
    "Dispositions: Battery": [
        1_338_039.1, 1_349_945.2, 1_820_033.5, 2_393_827.9, 2_205_778.5,
        2_309_624.0, 3_116_627.2, 3_353_703.1, 3_921_587.2, 4_282_688.2
    ],
    "Dispositions: Gas Gathering": [
        35_047_851.5, 35_128_406.0, 34_262784.7, 36_970_503.8, 39_301_946.0,
        39_230_714.2, 40_996_339.5, 44_517719.4, 45_293449.6, 46_917212.9],
    "Dispositions: Gas Plant": [
        28735752.6, 30791965.2, 35639543.6, 38395917.8, 37957098.2,
        37_564_708.5, 40_405_168.5, 41798958.6, 43022445.3, 43741498.3],
    "Disposition: Injection Facility": [
        3029359.0, 2354451.9, 2647212.6, 2362726.0, 2306028.8,
        3_451_790.3, 2_451_177.8, 2721800.6, 3917236.4, 3092869.2],
    "Dispositions: Meter Station": [
        163336155.9, 168123198.2, 171270090.7, 178072899.2, 173965630.9,
        165681431.1, 175_598_648.7, 184359220.1, 181595413.1, 192401140.5],
    "Disposition: Other": [
        16223.7, 13880.1, 13055.8, 25333.1, 105004.0,
        92_087.6, 118_026.1, 151677.0, 297708.3, 302511.3],
    #"Disposition: AB_Commercial":
    #"Disposition: AB_Industrial": 
    # "Disposition_AB_Residential":
    # "Disposition_AB_Elec_Gen": 
    "Dispositions: Shrinkage, Acid_Gas": [
        2407633.0, 2152474.8, 2001342.3, 1860785.7, 1777115.3,
        1_620_368.6, 1_458_068.9, 1394463.1, 1421915.3, 1286373.0,
    ],
    "Dispositions: Shrinkage, Other Products": [
        8301937.4, 8990144.3, 9381214.5, 9955474.6, 10495020.4,
        None, None, 11039508.9, 11174480.3, 11651783.2
    ],
    "Dispositions: Fuel": [
        5717123.7, 5771881.6, 5903623.9, 6017882.3, 6046326.3,
        None, None, 6104972.2, 6137404.9, 5984713.9
    ],
    "Dispositions: Flared": [
        180635.7, 144955.1, 165237.4, 185170.6, 227105.1,
        None, None, 492349.0, 439688.1, 440131.6
    ],
    "Dispositions: Vented": [
        10133.5, 10422.6, 8629.5, 7777.6, 8519.1,
        None, None, 93180.3, 96824.8, 110457.5
    ],
    "Process and Frac: Pentanes Plus": [
        8353776.1, 9828280.2, 11925102.0, 13317478.0, 14430264.4,
        None, None, 16797951.8, 17150140.5, 17888361.5
    ],
    "Process and Frac: Propane": [
        5373460.0, 6297459.1, 7601236.1, 8902503.4, 9843120.7,
        None, None, 10796410.0, 11013468.5, 11773388.7
    ],
    "Process and Frac: Butanes": [
        3976190.3, 4368459.7, 5027972.1, 5885288.1, 6205784.2,
        None, None, 7235699.6, 7159548.2, 7468735.2
    ],
    "Process and Frac: Ethane": [
        12968622.7, 13945157.9, 12971014.8, 13053605.3, 13253205.2,
        None, None, 12628888.6, 13203868.0, 13663822.0
    ],
    "Process and Frac: NGL": [
        16469345.4, 18455191.8, 20346208.4, 21320887.8, 22119744.3,
        None, None, 22899355.8, 23063399.3, 23578812.6
    ],
    "Process and Frac: Sulphur Production": [
        1950576.7, 1777227.3, 1694649.3, 1595942.6, 1525385.1,
        None, None, 1332156.0, 1281343.9, 1229707.3
    ],
    "Process and Frac: Sulphur Closing_Inventory": [
        863157.6, 820320.5, 794650.3, 581698.9, 665815.3,
        None, None, 804518.1, 732294.1, 544798.7
    ]
}

class BitumenProductionType(str, enum.Enum):
    InSitu = 'In Situ'
    Surface = 'Surface'


def st98_crude_bitumen_production():
    df = pd.read_excel(
        'data/st98-2025-crude-bitumen-supplydemand-data.xlsx',
        sheet_name="Figures",
        usecols='P:T',
        skiprows=33,
        nrows=37,
        names=['Year',
               'In situ (kilo m3/d)',
               'Surface mining (kilo m3/d)',
               'In situ (kilo bbl/d)',
               'Surface mining (kilo bbl/d)'])

    rval = objtensor.empty(BitumenProductionType)

    rval[BitumenProductionType.InSitu] = sts.annual_report2(
        years=df['Year'].values,
        values=df['In situ (kilo m3/d)'].values,
        v_unit=u.kilo_m3_crude_bitumen / u.day)
    rval[BitumenProductionType.Surface] = sts.annual_report2(
        years=df['Year'].values,
        values=df['Surface (kilo m3/d)'].values,
        v_unit=u.kilo_m3_crude_bitumen / u.day)
    return rval
