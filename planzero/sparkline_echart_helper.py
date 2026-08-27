import enum

import numpy as np

from . import nir2025
from .enums import IPCC_Sector, LULUCF_Sectors
from .html import (
    EChartDataZoomElem,
    EChartGrid,
    EChartItemStyle,
    EChartLineStyle,
    EChartMatrix,
    EChartMatrixBody,
    EChartMatrixBodyDataElem,
    EChartMatrixCorner,
    EChartMatrixXAxis,
    EChartMatrixXY,
    EChartMatrixYAxis,
    EChartSeriesBase,
    EChartToolTip,
    GridLinkElem,
    UncertainSparklineMatrixEChart,
)


class PseudoSectors(str, enum.Enum):
    Total_with_LULUCF = 'Total with LULUCF'
    Total_without_LULUCF = 'Total without LULUCF'


class SparklineEChartHelperBase:

    palette = (
        '#5470c6', '#91cc75', '#fac858', '#ee6666',
        '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
        '#ea7ccc', '#4A90E2', '#50E3C2', '#F5A623',
        '#D0021B', '#8B572A', '#417505', '#BD10E0'
    )

    n_non_lulucf_rows = 10
    n_total_rows = n_non_lulucf_rows + 2
    n_cols = 7

    credibility_interval_95 = (.025, .975)

    def __init__(self, div_id, v_unit, model_name):
        self.div_id = div_id
        self.model_name = model_name
        self.grid_list = []
        self.xAxis_list = []
        self.yAxis_list = []
        self.series_list = []
        self.grid_links = []
        self.v_unit = v_unit

        # has to be every year or else scaling doesn't work properly
        # when combined with historic actuals
        self.years = np.arange(1990, 2050 + 1)

        self.arr_pt, self.arr_ca = nir2025.ktCO2e_dense_w_nan()

        self.data_by_sector = {} # real sector and pseudo-sector

    @property
    def v_unit_scale(self) -> float:
        if self.v_unit == 'Mt_CO2e':
            return 0.001
        elif self.v_unit == 'kt_CO2e':
            return 1
        else:
            raise NotImplementedError(self.v_unit)

    def add_data_for_sector(self, sector, sector_means, lbounds, ubounds):
        assert sector not in self.data_by_sector
        assert np.all(ubounds >= lbounds)
        self.data_by_sector[sector] = {
            'ubound': np.max(ubounds),
            'lbound': np.min(lbounds),
            'means': sector_means,
            'ubounds': ubounds,
            'lbounds': lbounds,
            'CIs': ubounds - lbounds,
            'neg_shift': np.minimum(ubounds, 0),
            'neg_shade': np.minimum(lbounds, 0) - np.minimum(ubounds, 0),
            'pos_shift': np.maximum(lbounds, 0),
            'pos_shade': np.maximum(ubounds, 0) - np.maximum(lbounds, 0),
            }

    def compute_stats_and_add_data_for_sector(self, sector, sample):
        lbounds, ubounds = np.quantile(
            sample,
            q=self.credibility_interval_95,
            axis=0)

        mean_sector_total = np.mean(sample, axis=0)
        self.add_data_for_sector(
            sector,
            mean_sector_total,
            lbounds=lbounds,
            ubounds=ubounds)
        return mean_sector_total

    def add_data_for_LULUCF_totals(
        self,
        estimates_with_lulucf,
        mean_with_lulucf,
        estimates_without_lulucf,
        mean_without_lulucf,
        ):
        lbounds_with_lulucf, ubounds_with_lulucf = np.quantile(
            estimates_with_lulucf,
            q=self.credibility_interval_95,
            axis=0)

        self.add_data_for_sector(
            PseudoSectors.Total_with_LULUCF,
            mean_with_lulucf,
            lbounds=lbounds_with_lulucf,
            ubounds=ubounds_with_lulucf)

        lbounds_without_lulucf, ubounds_without_lulucf = np.quantile(
            estimates_without_lulucf,
            q=self.credibility_interval_95,
            axis=0)

        self.add_data_for_sector(
            PseudoSectors.Total_without_LULUCF,
            mean_without_lulucf,
            lbounds=lbounds_without_lulucf,
            ubounds=ubounds_without_lulucf)

    def order_sectors(self):

        # order sectors by decreasing last-year uncertainty
        non_lulucf_scores = [
            (-self.data_by_sector[sector]['ubound'], sector)
            for sector in IPCC_Sector
            if sector not in LULUCF_Sectors]
        non_lulucf_scores.sort()
        self.sorted_non_lulucf = [sector for _, sector in non_lulucf_scores]
        self.sorted_lulucf = [
            IPCC_Sector.Forest_Land,
            IPCC_Sector.Harvested_Wood_Products,
            IPCC_Sector.Settlements,
            IPCC_Sector.Cropland,
            IPCC_Sector.Wetlands,
            IPCC_Sector.Grassland,
        ]
        assert len(self.sorted_lulucf) == len(LULUCF_Sectors)

    def body_data(self):
        fontSize = 9
        rval = []
        rval.append(EChartMatrixBodyDataElem(
            coord=[0, 0],
            value='Total without LULUCF',
            label={'color': '#999', 'fontSize': fontSize, 'position': 'insideTop'},
            ))
        for row in range(self.n_non_lulucf_rows):
            for col in range(self.n_cols):
                if row == col == 0:
                    continue
                try:
                    sector = self.sorted_non_lulucf[row * self.n_cols + col - 1]
                except IndexError:
                    break
                rval.append(
                    EChartMatrixBodyDataElem(
                        coord=[col, row],
                        value=(sector.value
                               .replace('anufacturing', 'fg.')
                               .replace('roduction', 'rod.')
                               .replace('onsumption', 'ons.')
                               .replace('and Solvent Use', ', Solvents')
                               .replace('Carbon-Containing', '')
                              ),
                        label={
                            'color': '#999',
                            'fontSize': fontSize,
                            'position': 'insideTop'
                            },
                        )
                    )

        # merge the second-last row
        rval.append(
            EChartMatrixBodyDataElem(
                coord=[None, self.n_total_rows - 2],
                value='Land-Use, Land-Use Change, and Forestry (LULUCF)',
                coordClamp=True,
                mergeCells=True,
                label={'color': '#999', 'fontSize': 14},
                )
            )

        rval.append(EChartMatrixBodyDataElem(
            coord=[0, self.n_total_rows - 1],
            value='Total with LULUCF',
            label={'color': '#999', 'fontSize': fontSize, 'position': 'insideTop'},
            ))

        assert self.n_cols >= len(LULUCF_Sectors) + 1
        for col_minus_1, sector in enumerate(self.sorted_lulucf):
            rval.append(
                EChartMatrixBodyDataElem(
                    coord=[col_minus_1 + 1, self.n_total_rows - 1],
                    value=sector.value,
                    label={'color': '#999', 'fontSize': fontSize, 'position': 'insideTop'},
                    )
                )
        return rval

    def row_minmax(self, sectors):
        row_ymax = -float('inf')
        row_ymin = float('inf')
        for sector in sectors:
            data = self.data_by_sector[sector]
            row_ymax = max(row_ymax, max(data['ubounds']))
            row_ymin = min(row_ymin, min(data['lbounds']))

            if 'Total' in sector:
                actuals = np.sum(self.arr_ca, axis=(0, 1)) * self.v_unit_scale
            else:
                actuals = np.sum(self.arr_ca[nir2025.idx_of_sector[sector]], axis=0) * self.v_unit_scale
            row_ymax = max(row_ymax, np.nanmax(actuals))
            row_ymin = min(row_ymin, np.nanmin(actuals))

        # round up to nearest 2-significant-digit number
        row_ymax = max(0, float(f'{row_ymax * 1.06:.2g}'))
        row_ymin = min(0, float(f'{row_ymin * 1.06:.2g}'))
        return row_ymin, row_ymax

    def append_cell(self, row, col, sector, ymin, ymax, yAxis_customValues=None):
        #color = self.palette[(row * self.n_cols + col - 1) % len(self.palette)]
        if sector in LULUCF_Sectors:
            color = self.palette[9]
        elif sector == PseudoSectors.Total_without_LULUCF:
            color = self.palette[10]
        elif sector == PseudoSectors.Total_with_LULUCF:
            color = self.palette[11]
        else:
            color = {
                # energy - stationary
                'Stationary_Combustion_Sources': self.palette[0],
                # energy - transport
                'Transport': self.palette[1],
                'CO2_Transport_and_Storage': self.palette[1],
                # extraction - fugitive
                'Fugitive_Sources': self.palette[7],
                # industrial
                'Mineral_Products': self.palette[3],
                'Chemical_Industry': self.palette[4],
                'Metal_Production': self.palette[5],
                'Production_and_Consumption_of_Halocarbons,_SF6_and_NF3': self.palette[6],
                'Non-Energy_Products_from_Fuels_and_Solvent_Use': self.palette[6],
                'Other_Product_Manufacture_and_Use': self.palette[6],
                # agriculture
                'Enteric_Fermentation': self.palette[2],
                'Manure_Management': self.palette[2],
                'Agricultural_Soils': self.palette[2],
                'Field_Burning_of_Agricultural_Residues': self.palette[2],
                'Liming,_Urea_Application_and_Other_Carbon-Containing_Fertilizers': self.palette[2],
                # waste
                'Municipal_Solid_Waste_Landfills': self.palette[8],
                'Industrial_Wood_Waste_Lanfills': self.palette[8], # known typo
                'Biological_Treatment_of_Solid_Waste': self.palette[8],
                'Incineration_and_Open_Burning_of_Waste': self.palette[8],
                'Municipal_Wastewater_Treatment_and_Discharge': self.palette[8],
                'Industrial_Wastewater_and_Discharge': self.palette[8],
            }[sector.catpath_no_whitespace.split('/')[0]]

        self.grid_list.append(
            EChartGrid(
                id=f'grid_{col}|{row}',
                coordinateSystem='matrix',
                coord=[col, row],
                top=25,
                bottom=10,
                left='center',
                width='90%',
                containLabel=True,
                ))
        if sector in IPCC_Sector:
            self.grid_links.append(
                GridLinkElem(
                    gridId=f'grid_{col}|{row}',
                    url=f'/models/prob/{self.model_name}/sectors/{sector.catpath_no_whitespace}',
                    ))
        self.xAxis_list.append(
            EChartMatrixXAxis(
                type='category',
                id=f'xAxis_{col}|{row}',
                gridId=f'grid_{col}|{row}',
                scale=True,
                axisTick={'show': False},
                axisLabel={'show': False},
                axisLine={'show': False},
                splitLine={'show': False},
                boundaryGap=False,
                ))
        self.yAxis_list.append(
            EChartMatrixYAxis(
                id=f'yAxis_{col}|{row}',
                gridId=f'grid_{col}|{row}',
                interval=1_000_000_000_000, # ensure just two ticks per axis
                scale=True,
                max=ymax,
                min=ymin,
                axisLabel={'showMaxLabel': True,
                               'fontSize': 9,
                               'customValues': yAxis_customValues,
                              },
                axisLine={'show': False},
                axisTick={'show': False},
                ))

        # historical actuals
        if hasattr(self, 'nir2025_sparkline_echart_helper'):
            nir2025_means = self.nir2025_sparkline_echart_helper.data_by_sector[sector]['means']
            nir2025_lbounds = self.nir2025_sparkline_echart_helper.data_by_sector[sector]['lbounds']
            nir2025_ubounds = self.nir2025_sparkline_echart_helper.data_by_sector[sector]['ubounds']
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} NIR2025',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(
                        width=2,
                        type='dotted',
                        color='#000'),
                    data=list(zip(nir2025.nir2025_year_ints, nir2025_means)),
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} NIR2025',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(
                        width=1,
                        type='solid',
                        color='#000'),
                    data=list(zip(nir2025.nir2025_year_ints, nir2025_lbounds)),
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} NIR2025',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(
                        width=1,
                        type='solid',
                        color='#000'),
                    data=list(zip(nir2025.nir2025_year_ints, nir2025_ubounds)),
                    ))
        else:
            if sector == PseudoSectors.Total_without_LULUCF:
                mask = [(sec not in LULUCF_Sectors) for sec in IPCC_Sector]
                actuals = np.sum(self.arr_ca[mask], axis=(0, 1))
            elif sector == PseudoSectors.Total_with_LULUCF:
                actuals = np.sum(self.arr_ca, axis=(0, 1))
            else:
                actuals = np.sum(self.arr_ca[nir2025.idx_of_sector[sector]], axis=0)

            actuals = actuals * self.v_unit_scale

            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} NIR2025',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(
                        width=2,
                        color='#000'),
                    data=list(zip(nir2025.nir2025_year_ints, actuals)),
                    ))

        data = self.data_by_sector[sector]
        self.series_list.append(
            EChartSeriesBase(
                name=f'{sector} mean',
                xAxisId=f'xAxis_{col}|{row}',
                yAxisId=f'yAxis_{col}|{row}',
                type='line',
                symbol='none',
                lineStyle=EChartLineStyle(
                    width=2,
                    type='dotted',
                    color=color),
                data=list(zip(self.years, data['means'])),
                ))
        if data['lbound'] < 0:
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} CI lower bound',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0, color=color),
                    data=list(zip(self.years, data['neg_shift'])),
                    stack=f'stack_{sector:s}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle={'opacity': .25},
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['neg_shade'])),
                    stack=f'stack_{sector:s}'
                    ))
        if data['ubound'] >= 0:
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} CI lower bound',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0, color=color),
                    data=list(zip(self.years, data['pos_shift'])),
                    stack=f'stack_{sector:s}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle={'opacity': .25},
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['pos_shade'])),
                    stack=f'stack_{sector:s}'
                    ))

    def add_non_lulucf_cells(self):
        for row in range(self.n_non_lulucf_rows):
            sectors = self.sorted_non_lulucf[
                max(0, row * self.n_cols - 1):
                (row + 1) * self.n_cols - 1]
            row_ymin, row_ymax = self.row_minmax(sectors)
            for col in range(self.n_cols):
                if row == col == 0:
                    continue
                try:
                    sector = self.sorted_non_lulucf[row * self.n_cols + col - 1]
                except IndexError:
                    break
                self.append_cell(row, col, sector, row_ymin, row_ymax)

    def add_lulucf_cells(self):
        assert self.n_cols >= len(LULUCF_Sectors) + 1
        row_ymin, row_ymax = self.row_minmax(self.sorted_lulucf)
        for col_minus_1, sector in enumerate(self.sorted_lulucf):
            self.append_cell(self.n_total_rows - 1,
                             col_minus_1 + 1,
                             sector, ymin=row_ymin,
                             ymax=row_ymax)

    def add_total_cells(self):
        self.append_cell(0, 0,
                         sector=PseudoSectors.Total_without_LULUCF,
                         ymin=0,
                         ymax=None)
        self.append_cell(self.n_total_rows - 1, 0,
                         sector=PseudoSectors.Total_with_LULUCF,
                         ymin=0,
                         ymax=None)

    def make_echart(self):
        rval = UncertainSparklineMatrixEChart(
            div_id=self.div_id,
            matrix=EChartMatrix(
                x=EChartMatrixXY(
                    length=self.n_cols,
                    levelSize=40,
                    show=False,
                    ),
                y=EChartMatrixXY(
                    length=self.n_total_rows,
                    levelSize=80,
                    show=False,
                    ),
                corner=EChartMatrixCorner(data=[], label={}),
                body=EChartMatrixBody(
                    data=self.body_data()),
                top=30,
                bottom=80,
                width='95%',
                left='center',
                ),
            tooltip=EChartToolTip(trigger='axis'),
            dataZoom=[
                EChartDataZoomElem(
                    type='slider',
                    xAxisIndex='all',
                    left='10%',
                    right='10%',
                    bottom=30,
                    height=30,
                    throttle=120,
                    ),
                EChartDataZoomElem(
                    type='inside',
                    xAxisIndex='all',
                    throttle=120,
                    ),
                ],
            grid=self.grid_list,
            xAxis=self.xAxis_list,
            yAxis=self.yAxis_list,
            series=self.series_list,
            grid_links=self.grid_links,
            width='100%',
            height='1000px',
            )
        return rval
