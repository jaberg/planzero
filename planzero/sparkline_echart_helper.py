import enum

import numpy as np

from . import nir2025
from .enums import GHG, PT, IPCC_Sector, LULUCF_Sectors, col_by_pt, col_ca
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


# TODO: if enums.Regions becomes a thing,
# then no need for this anymore
class PseudoRegion(str, enum.Enum):
    NationalTotal = 'National Total'


class BaseBase:

    n_cols = 7

    credibility_interval_95 = (.025, .975)

    def __init__(self, div_id, model_name):
        self.div_id = div_id
        self.model_name = model_name
        self.grid_list = []
        self.xAxis_list = []
        self.yAxis_list = []
        self.series_list = []
        self.grid_links = []
        self.stats_d = {} # key -> stats e.g. lbounds, ubounds, etc.
        self.sorted_keys = [] # list of keys in raster order of panels
        self.color_by_key = {}

        # has to be every year or else scaling doesn't work properly
        # when combined with historic actuals
        self.years = np.arange(1990, 2050 + 1)

    def update_stats_from_means_bounds(self, key, means, lbounds, ubounds):
        assert np.all(ubounds >= lbounds)
        self.stats_d[key] = {
            'ubound': np.max(ubounds),
            'lbound': np.min(lbounds),
            'means': means,
            'ubounds': ubounds,
            'lbounds': lbounds,
            'CIs': ubounds - lbounds,
            'neg_shift': np.minimum(ubounds, 0),
            'neg_shade': np.minimum(lbounds, 0) - np.minimum(ubounds, 0),
            'pos_shift': np.maximum(lbounds, 0),
            'pos_shade': np.maximum(ubounds, 0) - np.maximum(lbounds, 0),
        }
        self.stats_d[key]['spread'] = (
                self.stats_d[key]['ubound']
                - self.stats_d[key]['lbound'])

    def update_stats_from_sample(self, key, sample):
        lbounds, ubounds = np.quantile(
            sample,
            q=self.credibility_interval_95,
            axis=0)
        return self.update_stats_from_means_bounds(
                key,
                means=np.mean(sample, axis=0),
                lbounds=lbounds,
                ubounds=ubounds)

    def determine_n_total_rows(self):
        n_panels = len(self.sorted_keys)
        n_rows = 0
        while n_rows * self.n_cols < n_panels:
            n_rows += 1
        self.n_total_rows = n_rows

    def row_minmax(self, keys):
        row_ymax = -float('inf')
        row_ymin = float('inf')
        for key in keys:
            stats = self.stats_d[key]
            row_ymax = max(row_ymax, max(stats['ubounds']))
            row_ymin = min(row_ymin, min(stats['lbounds']))

        # round up to nearest 2-significant-digit number
        row_ymax = max(0, float(f'{row_ymax * 1.06:.2g}'))
        row_ymin = min(0, float(f'{row_ymin * 1.06:.2g}'))
        return row_ymin, row_ymax

    def append_cell_grid_and_axes(self, row, col, ymin, ymax):
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

        xaxis_id = f'xAxis_{col}|{row}'
        self.xAxis_list.append(
            EChartMatrixXAxis(
                type='category',
                id=xaxis_id,
                gridId=f'grid_{col}|{row}',
                scale=True,
                axisTick={'show': False},
                axisLabel={'show': False},
                axisLine={'show': False},
                splitLine={'show': False},
                boundaryGap=False,
                ))

        yaxis_id = f'yAxis_{col}|{row}'
        self.yAxis_list.append(
            EChartMatrixYAxis(
                id=yaxis_id,
                gridId=f'grid_{col}|{row}',
                interval=1_000_000_000_000, # ensure just two ticks per axis
                scale=True,
                max=ymax,
                min=ymin,
                axisLabel={
                    'showMaxLabel': True,
                    'fontSize': 9,
                    'customValues': None,
                    },
                axisLine={'show': False},
                axisTick={'show': False},
                ))
        return xaxis_id, yaxis_id

    def assign_default_colors(self):
        from .enums import echarts_warm_earth
        for ii, key in enumerate(self.sorted_keys):
            default_color = echarts_warm_earth[ii % len(echarts_warm_earth)]
            self.color_by_key.setdefault(key, default_color)

    def append_cell_data(self, key, xaxis_id, yaxis_id):
        data = self.stats_d[key]
        color = self.color_by_key[key]
        self.series_list.append(
            EChartSeriesBase(
                name=f'{key} mean',
                xAxisId=xaxis_id,
                yAxisId=yaxis_id,
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
                    name=(f'{key} CI upper bound'
                          if data['ubound'] < 0
                          else None),
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0, color=color),
                    data=list(zip(self.years, data['neg_shift'])),
                    stack=f'stack_{key!s}_neg'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=(f'{key} CI width (negative region)'
                          if data['ubound'] < 0
                          else f'{key} CI lower bound'),
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle=dict(opacity=.25),
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['neg_shade'])),
                    stack=f'stack_{key!s}_neg'
                    ))
        if data['ubound'] >= 0:
            self.series_list.append(
                EChartSeriesBase(
                    name=(f'{key} CI lower bound'
                          if data['lbound'] >= 0
                          else None),
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0, color=color),
                    data=list(zip(self.years, data['pos_shift'])),
                    stack=f'stack_{key!s}_pos'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=(f'{key} CI width'
                          if data['lbound'] >= 0
                          else f'{key} CI upper bound'),
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle=dict(opacity=.25),
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['pos_shade'])),
                    stack=f'stack_{key!s}_pos'
                    ))

    def append_cell(self, row, col, key, row_ymin, row_ymax):
        xaxis_id, yaxis_id = self.append_cell_grid_and_axes(
                row, col, row_ymin, row_ymax)
        self.append_cell_data(key, xaxis_id, yaxis_id)

    def append_all_cells(self):
        for row in range(self.n_total_rows):
            row_keys = self.sorted_keys[
                row * self.n_cols:
                (row + 1) * self.n_cols]
            assert 0 < len(row_keys) <= self.n_cols
            row_ymin, row_ymax = self.row_minmax(row_keys)
            for col, key in enumerate(row_keys):
                self.append_cell(row, col, key, row_ymin, row_ymax)

    def body_data(self):
        fontSize = 9
        rval = []
        for row in range(self.n_total_rows):
            for col in range(self.n_cols):
                try:
                    key = self.sorted_keys[row * self.n_cols + col]
                except IndexError:
                    break
                rval.append(
                    EChartMatrixBodyDataElem(
                        coord=[col, row],
                        value=key.value,
                        label={
                            'color': '#999',
                            'fontSize': fontSize,
                            'position': 'insideTop'},
                        )
                    )
        return rval

    def make_echart(self):
        height = {
                1: 225,
                2: 400,
                }[self.n_total_rows]
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
            width='100%',
            height=f'{height}px',
            )
        return rval


class SparklineEChartHelperBase(BaseBase):

    palette = (
        '#5470c6', '#91cc75', '#fac858', '#ee6666',
        '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
        '#ea7ccc', '#4A90E2', '#50E3C2', '#F5A623',
        '#D0021B', '#8B572A', '#417505', '#BD10E0'
    )

    n_non_lulucf_rows = 10
    n_total_rows = n_non_lulucf_rows + 2
    cells_include_actuals = True

    @property
    def v_unit_scale(self) -> float:
        if self.v_unit == 'Mt_CO2e':
            return 0.001
        elif self.v_unit == 'kt_CO2e':
            return 1
        else:
            raise NotImplementedError(self.v_unit)

    def __init__(self, div_id, v_unit, model_name):
        super().__init__(div_id=div_id, model_name=model_name)
        self.data_by_sector = {}
        self.data_by_sector = self.stats_d
        self.v_unit = v_unit

        self.arr_pt, self.arr_ca = nir2025.ktCO2e_dense_w_nan()

    def add_data_for_sector(self, sector, sector_means, lbounds, ubounds):
        assert sector not in self.data_by_sector
        return self.update_stats_from_means_bounds(
                sector, sector_means, lbounds, ubounds)

    def compute_stats_and_add_data_for_sector(self, sector, sample):
        self.update_stats_from_sample(sector, sample)
        return self.stats_d[sector]['means']

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

        if self.n_total_rows > self.n_non_lulucf_rows:
            assert self.n_total_rows == self.n_non_lulucf_rows + 2

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

            if self.cells_include_actuals:
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

    def append_cell(self, row, col, sector, ymin, ymax):
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
                               'customValues': None,
                              },
                axisLine={'show': False},
                axisTick={'show': False},
                ))

        if self.cells_include_actuals:
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
        key = sector
        if data['lbound'] < 0:
            self.series_list.append(
                EChartSeriesBase(
                    name=(f'{key} CI upper bound'
                          if data['ubound'] < 0
                          else None),
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
                    name=(f'{key} CI width (negative region)'
                          if data['ubound'] < 0
                          else f'{key} CI lower bound'),
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
                    name=(f'{key} CI lower bound'
                          if data['lbound'] >= 0
                          else None),
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
                    name=(f'{key} CI width'
                          if data['lbound'] >= 0
                          else f'{key} CI upper bound'),
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

    def add_total_cells(self, ymin_without_lulucf=0):
        self.append_cell(0, 0,
                         sector=PseudoSectors.Total_without_LULUCF,
                         ymin=ymin_without_lulucf,
                         ymax=None)
        if self.sorted_lulucf:
            self.append_cell(self.n_total_rows - 1, 0,
                             sector=PseudoSectors.Total_with_LULUCF,
                             ymin=0,
                             ymax=None)

    def make_echart(self):
        if self.n_total_rows == 1:
            height = 200
        elif self.n_total_rows == 2:
            raise NotImplementedError()
        else:
            height = 1000
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
            height=f'{height}px',
            )
        return rval


class RegionalSparklineEChartHelperBase(BaseBase):

    """
    Build a minigrid for a single sector, highlighting regional differences.
    It will have tiles for regions, arranged as 2 x 7.
    It may be a sum across GHGs or just one GHG
    """

    n_rows = 2
    n_cols = 7

    @property
    def v_unit_scale(self) -> float:
        if self.v_unit == 'Mt_CO2e':
            return 0.001
        elif self.v_unit == 'kt_CO2e':
            return 1
        else:
            raise NotImplementedError(self.v_unit)


    def __init__(self, sector, ghg:GHG|None, div_id, v_unit):
        super().__init__(div_id=div_id, model_name=None)
        self.sector = sector
        self.ghg = ghg
        self.data_by_region = self.stats_d
        self.v_unit = v_unit

        self.arr_pt, self.arr_ca = nir2025.ktCO2e_dense_w_nan()
        self.color_by_key[PseudoRegion.NationalTotal] = col_ca
        self.color_by_key.update(col_by_pt)

    def add_data_for_region(self, region, means, lbounds, ubounds):
        assert region not in self.data_by_region
        self.update_stats_from_means_bounds(region, means, lbounds, ubounds)

    def add_data_from_estimates(self, estimates_pt, estimates_ca):
        for ii, pt in enumerate(PT):
            if pt == PT.XX:
                continue
            pt_mean = np.mean(estimates_pt[:, :, ii], axis=0)
            pt_lbound, pt_ubound = np.quantile(
                estimates_pt[:, :, ii],
                self.credibility_interval_95,
                axis=0)
            self.add_data_for_region(pt, pt_mean, pt_lbound, pt_ubound)

        ca_mean = np.mean(estimates_ca, axis=0)
        ca_lbound, ca_ubound = np.quantile(
            estimates_ca,
            self.credibility_interval_95,
            axis=0)
        self.add_data_for_region(PseudoRegion.NationalTotal,
                                 ca_mean, ca_lbound, ca_ubound)

    def order_regions(self):

        # order sectors by decreasing last-year uncertainty
        scores = [
            (-self.data_by_region[region]['ubound'], region)
            for region in PT
            if region != PT.XX]
        scores.sort()
        _, self.sorted_keys = zip(*scores)
        self.sorted_keys = [PseudoRegion.NationalTotal] + list(self.sorted_keys)

    def actuals_in_v_unit_scale(self, region):
        if 'National' in region:
            if self.ghg:
                actuals = (
                    self.arr_ca[nir2025.idx_of_sector[self.sector],
                                nir2025.idx_of_ghg[self.ghg]]
                    * self.v_unit_scale)
            else:
                actuals = (
                    np.sum(self.arr_ca[nir2025.idx_of_sector[self.sector]], axis=0)
                    * self.v_unit_scale)
        else:
            if self.ghg:
                actuals = (
                    self.arr_pt[nir2025.idx_of_sector[self.sector],
                                nir2025.idx_of_ghg[self.ghg],
                                nir2025.idx_of_pt[region]]
                    * self.v_unit_scale)
            else:
                actuals = (
                    np.sum(self.arr_pt[nir2025.idx_of_sector[self.sector],
                                       :, 
                                       nir2025.idx_of_pt[region]], axis=0)
                    * self.v_unit_scale)
        try:
            assert len(actuals), (region, self.sector, self.ghg)
        except TypeError:
            assert 0, (region, self.sector, self.ghg)
        return actuals

    def row_minmax(self, regions):
        row_ymax = -float('inf')
        row_ymin = float('inf')
        for region in regions:
            data = self.data_by_region[region]
            row_ymax = max(row_ymax, max(data['ubounds']))
            row_ymin = min(row_ymin, min(data['lbounds']))

            actuals = self.actuals_in_v_unit_scale(region)
            row_ymax = max(row_ymax, np.nanmax(actuals))
            row_ymin = min(row_ymin, np.nanmin(actuals))

        # round up to nearest 2-significant-digit number
        row_ymax = max(0, float(f'{row_ymax * 1.06:.2g}'))
        row_ymin = min(0, float(f'{row_ymin * 1.06:.2g}'))
        return row_ymin, row_ymax

    def append_cell(self, row, col, region, row_ymin, row_ymax):
        xaxis_id, yaxis_id = self.append_cell_grid_and_axes(
                row, col, row_ymin, row_ymax)

        if hasattr(self, 'nir2025_regional_sparkline_echart_helper'):
            nir2025_means = self.nir2025_regional_sparkline_echart_helper.data_by_region[region]['means']
            nir2025_lbounds = self.nir2025_regional_sparkline_echart_helper.data_by_region[region]['lbounds']
            nir2025_ubounds = self.nir2025_regional_sparkline_echart_helper.data_by_region[region]['ubounds']
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} NIR2025',
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
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
                    name=f'{region} NIR2025',
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
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
                    name=f'{region} NIR2025',
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(
                        width=1,
                        type='solid',
                        color='#000'),
                    data=list(zip(nir2025.nir2025_year_ints, nir2025_ubounds)),
                    ))
        else:
            actuals = self.actuals_in_v_unit_scale(region)

            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} NIR2025',
                    xAxisId=xaxis_id,
                    yAxisId=yaxis_id,
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(
                        width=2,
                        color='#000'),
                    data=list(zip(nir2025.nir2025_year_ints, actuals)),
                    ))
        self.append_cell_data(region, xaxis_id, yaxis_id)


    def add_regional_cells(self):
        self.n_total_rows = self.n_rows
        self.append_all_cells()


from .annual_subsidy_results import NationalAnnualProgramBalances
from .enums import GovernmentProgram


def echart_from_napb(
        napb:NationalAnnualProgramBalances,
        div_id:str,
        model_name:str, # for linking charts to /models/prob/{model_name}/
        ):
    helper = BaseBase(
            div_id=div_id,
            model_name=model_name,
            )
    assert GovernmentProgram.Net in napb.CAD_sample
    for program, balance_sample in napb.CAD_sample.items():
        helper.update_stats_from_sample(program, balance_sample)
    non_net_programs = [
            prog for prog in napb.CAD_sample
            if prog != GovernmentProgram.Net]
    non_net_programs.sort(key=lambda prog: helper.stats_d[prog]['spread'])
    helper.sorted_keys.append(GovernmentProgram.Net)
    helper.sorted_keys.extend(non_net_programs)
    helper.determine_n_total_rows()
    helper.color_by_key = {
        GovernmentProgram.Net: col_ca,
    }
    helper.assign_default_colors()
    helper.append_all_cells()
    return helper.make_echart()
