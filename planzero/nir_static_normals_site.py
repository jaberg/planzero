import datetime
import enum

import numpy as np

from .enums import IPCC_Sector, GHG, LULUCF_Sectors, col_by_pt, col_ca
from .html import (
    UncertainSparklineMatrixEChart,
    EChartMatrix,
    EChartMatrixBody,
    EChartMatrixBodyDataElem,
    EChartMatrixCorner,
    EChartMatrixXY,
    EChartMatrixXAxis,
    EChartMatrixYAxis,
    EChartToolTip,
    EChartDataZoomElem,
    EChartGrid,
    EChartSeriesBase,
    EChartLineStyle,
    EChartItemStyle,
    GridLinkElem,
    )
from .nir_static_normals import (
    inference_work_loop,
    model_id_from_data_cutoff,
    touch_model,
    touch_components)

from .prob import SiteInference, ClassVar, computed_field


class PseudoSectors(str, enum.Enum):
    Total_with_LULUCF = 'Total with LULUCF'
    Total_without_LULUCF = 'Total without LULUCF'


class SparklineEChartHelper(object):

    palette = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666',
        '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
        '#ea7ccc', '#4A90E2', '#50E3C2', '#F5A623',
        '#D0021B', '#8B572A', '#417505', '#BD10E0'
    ]

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

        # has to be every year or else scaling doesn't work properly
        # when combined with historic actuals
        self.years = np.arange(1990, 2050+1)

        self.arr_pt, self.arr_ca = nir2025.ktCO2e_dense_w_nan()

        self.data_by_sector = {} # real sector and pseudo-sector
        if v_unit == 'Mt_CO2e':
            self.v_unit_scale = 0.001
        elif v_unit == 'kt_CO2e':
            self.v_unit_scale = 1
        else:
            raise NotImplementedError(v_unit)

    def add_data_for_sector(self, sector, sector_mean, lbound, ubound):
        assert sector not in self.data_by_sector
        assert ubound >= lbound
        self.data_by_sector[sector] = dict(
            mean=sector_mean,
            ubound=ubound,
            lbound=lbound,
            CI=ubound - lbound,
            means=[sector_mean for yr in self.years],
            ubounds=[ubound for yr in self.years],
            lbounds=[lbound for yr in self.years],
            CIs=[ubound - lbound for yr in self.years],
            neg_shift=[min(ubound, 0) for yr in self.years],
            neg_shade=[min(lbound, 0) - min(ubound, 0) for yr in self.years],
            pos_shift=[max(lbound, 0) for yr in self.years],
            pos_shade=[max(ubound, 0) - max(lbound, 0) for yr in self.years],
            )

    def load_data(self):
        self.config = load_config(allow_version_mismatch=False)

        n_samples = 100 # match saved data
        n_new_draws = 125

        mean_with_lulucf = 0
        estimates_with_lulucf = np.zeros((n_new_draws, n_samples))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros((n_new_draws, n_samples))

        for sector in IPCC_Sector:
            sector_mean = 0

            rng = np.random.default_rng(seed=123)
            estimates = rng.standard_normal((n_new_draws, n_samples, len(GHG)))

            for ii, ghg in enumerate(GHG):
                if [str(sector), str(ghg)] in self.config['near_zero_sector_ghgs']:
                    estimates[:, :, ii] = 0
                else:
                    config_sg, samples = load_config_samples(sector, ghg)
                    n_chains, n_samples_, n_regions = samples['mu'].shape
                    assert n_samples_ == n_samples
                    sector_mean_ghg = float(
                        samples['mu']
                        .reshape((n_chains * n_samples, n_regions))
                        .mean(axis=0) # across samples and chains
                        .sum(axis=0)) # over regions
                    sector_mean += sector_mean_ghg * config_sg['scale'] * self.v_unit_scale

                    estimates[:, :, ii] *= samples['sigma_ca']
                    estimates[:, :, ii] += samples['mu'].sum(axis=2)
                    estimates[:, :, ii] *= config_sg['scale'] * self.v_unit_scale

            sector_estimates = np.sum(estimates, axis=2)

            lbound, ubound = np.quantile(
                sector_estimates.flatten(),
                self.credibility_interval_95)

            self.add_data_for_sector(sector, sector_mean, lbound, ubound)

            if sector not in LULUCF_Sectors:
                estimates_without_lulucf += sector_estimates
                mean_without_lulucf += sector_mean
            estimates_with_lulucf += sector_estimates
            mean_with_lulucf += sector_mean

        lbound_with_lulucf, ubound_with_lulucf = np.quantile(
            estimates_with_lulucf.flatten(),
            self.credibility_interval_95)
        self.add_data_for_sector(
            PseudoSectors.Total_with_LULUCF,
            mean_with_lulucf,
            lbound_with_lulucf,
            ubound_with_lulucf)

        lbound_without_lulucf, ubound_without_lulucf = np.quantile(
            estimates_without_lulucf.flatten(),
            self.credibility_interval_95)
        self.add_data_for_sector(
            PseudoSectors.Total_without_LULUCF,
            mean_without_lulucf,
            lbound_without_lulucf,
            ubound_without_lulucf)


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
            label=dict(color='#999', fontSize=fontSize, position='insideTop'),
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
                        label=dict(color='#999',
                                   fontSize=fontSize,
                                   position='insideTop'),
                        )
                    )

        # merge the second-last row
        rval.append(
            EChartMatrixBodyDataElem(
                coord=[None, self.n_total_rows - 2],
                value='Land-Use, Land-Use Change, and Forestry (LULUCF)',
                coordClamp=True,
                mergeCells=True,
                label=dict(color='#999',
                           fontSize=14),
                )
            )

        rval.append(EChartMatrixBodyDataElem(
            coord=[0, self.n_total_rows - 1],
            value='Total with LULUCF',
            label=dict(color='#999', fontSize=fontSize, position='insideTop'),
            ))

        assert self.n_cols >= len(LULUCF_Sectors) + 1
        for col_minus_1, sector in enumerate(self.sorted_lulucf):
            rval.append(
                EChartMatrixBodyDataElem(
                    coord=[col_minus_1 + 1, self.n_total_rows - 1],
                    value=sector.value,
                    label=dict(color='#999',
                               fontSize=fontSize,
                               position='insideTop'),
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
                axisTick=dict(show=False),
                axisLabel=dict(show=False),
                axisLine=dict(show=False),
                splitLine=dict(show=False),
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
                axisLabel=dict(showMaxLabel=True,
                               fontSize=9,
                               customValues=yAxis_customValues,
                              ),
                axisLine=dict(show=False),
                axisTick=dict(show=False),
                ))

        # historical actuals
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
                    stack=f'stack_{str(sector)}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle=dict(opacity=.25),
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['neg_shade'])),
                    stack=f'stack_{str(sector)}'
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
                    stack=f'stack_{str(sector)}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{sector} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle=dict(opacity=.25),
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['pos_shade'])),
                    stack=f'stack_{str(sector)}'
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


class PseudoRegion(str, enum.Enum):
    NationalTotal = 'National Total'


class RegionalSparklineEChartHelper(object):

    """
    Build a minigrid for a single sector.
    It will have tiles for regions, arranged as 2 x 7.
    It may be a sum across GHGs or just one GHG
    """

    n_rows = 2
    n_cols = 7

    credibility_interval_95 = (.025, .975)

    def __init__(self, sector, ghg:GHG|None, div_id, v_unit):
        self.sector = sector
        self.ghg = ghg
        self.div_id = div_id
        self.grid_list = []
        self.xAxis_list = []
        self.yAxis_list = []
        self.series_list = []

        # has to be every year or else scaling doesn't work properly
        # when combined with historic actuals
        self.years = np.arange(1990, 2050+1)

        self.arr_pt, self.arr_ca = nir2025.ktCO2e_dense_w_nan()

        self.data_by_region = {} # real PT and pseudo-sector
        if v_unit == 'Mt_CO2e':
            self.v_unit_scale = 0.001
        elif v_unit == 'kt_CO2e':
            self.v_unit_scale = 1
        else:
            raise NotImplementedError(v_unit)

    def add_data_for_region(self, region, mean, lbound, ubound):
        assert region not in self.data_by_region
        assert ubound >= lbound
        self.data_by_region[region] = dict(
            mean=mean,
            ubound=ubound,
            lbound=lbound,
            CI=ubound - lbound,
            means=[mean for yr in self.years],
            ubounds=[ubound for yr in self.years],
            lbounds=[lbound for yr in self.years],
            CIs=[ubound - lbound for yr in self.years],
            neg_shift=[min(ubound, 0) for yr in self.years],
            neg_shade=[min(lbound, 0) - min(ubound, 0) for yr in self.years],
            pos_shift=[max(lbound, 0) for yr in self.years],
            pos_shade=[max(ubound, 0) - max(lbound, 0) for yr in self.years],
            )

    def load_data(self):
        self.config = load_config(allow_version_mismatch=False)

        n_samples = 100 # match saved data
        n_new_draws = 125

        rng = np.random.default_rng(seed=123)
        estimates_ghg_pt = rng.standard_normal((n_new_draws, n_samples, len(GHG), 13))
        estimates_ghg_ca = rng.standard_normal((n_new_draws, n_samples, len(GHG)))

        for ii, ghg in enumerate(GHG):
            if [str(self.sector), str(ghg)] in self.config['near_zero_sector_ghgs']:
                estimates_ghg_pt[:, :, ii] = 0
                estimates_ghg_ca[:, :, ii] = 0
            elif self.ghg is not None and self.ghg != ghg:
                estimates_ghg_pt[:, :, ii] = 0
                estimates_ghg_ca[:, :, ii] = 0
            else:
                config_sg, samples = load_config_samples(self.sector, ghg)
                n_chains, n_samples_, n_regions = samples['mu'].shape
                assert n_samples_ == n_samples

                estimates_ghg_pt[:, :, ii] *= samples['sigma_pt']
                estimates_ghg_pt[:, :, ii] += samples['mu']
                estimates_ghg_pt[:, :, ii] *= config_sg['scale'] * self.v_unit_scale

                estimates_ghg_ca[:, :, ii] *= samples['sigma_ca']
                estimates_ghg_ca[:, :, ii] += samples['mu'].sum(axis=2)
                estimates_ghg_ca[:, :, ii] *= config_sg['scale'] * self.v_unit_scale

        estimates_pt = estimates_ghg_pt.sum(axis=2)
        estimates_ca = estimates_ghg_ca.sum(axis=2)

        for ii, pt in enumerate(PT):
            if pt == PT.XX:
                continue
            pt_mean = np.mean(estimates_pt[:, :, ii])
            pt_lbound, pt_ubound = np.quantile(
                estimates_pt[:, :, ii].flatten(),
                self.credibility_interval_95)
            self.add_data_for_region(pt, pt_mean, pt_lbound, pt_ubound)

        ca_mean = np.mean(estimates_ca)
        ca_lbound, ca_ubound = np.quantile(
            estimates_ca.flatten(),
            self.credibility_interval_95)
        self.add_data_for_region(PseudoRegion.NationalTotal,
                                 ca_mean, ca_lbound, ca_ubound)

    def order_regions(self):

        # order sectors by decreasing last-year uncertainty
        scores = [
            (-self.data_by_region[region]['ubound'], region)
            for region in PT
            if region != PT.XX]
        scores.sort()
        _, self.sorted_regions = zip(*scores)
        self.sorted_regions = [PseudoRegion.NationalTotal] + list(self.sorted_regions)

    def body_data(self):
        fontSize = 9
        rval = []
        #rval.append(EChartMatrixBodyDataElem(
        #    coord=[0, 0],
        #    value='National Total',
        #    label=dict(color='#999', fontSize=fontSize, position='insideTop'),
        #    ))
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                try:
                    region = self.sorted_regions[row * self.n_cols + col]
                except IndexError:
                    break
                rval.append(
                    EChartMatrixBodyDataElem(
                        coord=[col, row],
                        value=(region.value),
                        label=dict(color='#999',
                                   fontSize=fontSize,
                                   position='insideTop'),
                        )
                    )
        return rval

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

    def color_of_region(self, region):
        if region == PseudoRegion.NationalTotal:
            color = col_ca
        else:
            color = col_by_pt[region]
        return color

    def append_cell(self, row, col, region, ymin, ymax, yAxis_customValues=None):
        color = self.color_of_region(region)

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
        self.xAxis_list.append(
            EChartMatrixXAxis(
                type='category',
                id=f'xAxis_{col}|{row}',
                gridId=f'grid_{col}|{row}',
                scale=True,
                axisTick=dict(show=False),
                axisLabel=dict(show=False),
                axisLine=dict(show=False),
                splitLine=dict(show=False),
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
                axisLabel=dict(showMaxLabel=True,
                               fontSize=9,
                               customValues=yAxis_customValues,
                              ),
                axisLine=dict(show=False),
                axisTick=dict(show=False),
                ))

        actuals = self.actuals_in_v_unit_scale(region)

        self.series_list.append(
            EChartSeriesBase(
                name=f'{region} NIR2025',
                xAxisId=f'xAxis_{col}|{row}',
                yAxisId=f'yAxis_{col}|{row}',
                type='line',
                symbol='none',
                lineStyle=EChartLineStyle(
                    width=2,
                    color='#000'),
                data=list(zip(nir2025.nir2025_year_ints, actuals)),
                ))

        data = self.data_by_region[region]
        self.series_list.append(
            EChartSeriesBase(
                name=f'{region} mean',
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
                    name=f'{region} CI lower bound',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0, color=color),
                    data=list(zip(self.years, data['neg_shift'])),
                    stack=f'stack_{str(region)}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle=dict(opacity=.25),
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['neg_shade'])),
                    stack=f'stack_{str(region)}'
                    ))
        if data['ubound'] >= 0:
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} CI lower bound',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0, color=color),
                    data=list(zip(self.years, data['pos_shift'])),
                    stack=f'stack_{str(region)}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle=dict(opacity=.25),
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['pos_shade'])),
                    stack=f'stack_{str(region)}'
                    ))

    def add_regional_cells(self):
        for row in range(self.n_rows):
            row_regions = self.sorted_regions[
                row * self.n_cols:
                (row + 1) * self.n_cols]
            row_ymin, row_ymax = self.row_minmax(row_regions)
            for col in range(self.n_cols):
                try:
                    region = row_regions[col]
                except IndexError:
                    break
                self.append_cell(row, col, region, row_ymin, row_ymax)

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
                    length=self.n_rows,
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
            height='400px',
            )
        return rval


class Static_Normals_2024_12_31(SiteInference):
    """Emissions per province and territory,
    and per greenhouse gas, modelled as
    non-time-varying Normal distributions.
    National totals are modelled as the sums of provincial
    and territorial totals.
    This is a baseline model, not intended to be accurate.
    """

    include_in_registry: ClassVar[bool] = True

    data_cutoff:object = datetime.date(year=2024, month=12, day=31)

    def main_model_id(self):
        model_id = model_id_from_data_cutoff(self.data_cutoff)
        print(model_id)

    def main_inference_prep(self):
        touch_model(self.data_cutoff)
        touch_components(self.data_cutoff)

    def main_inference_work(self):
        # this entrypoint may be used by concurrent workers
        inference_work_loop(self.data_cutoff)

    def one_line_description(self):
        return "Very simple baseline model - non-time-varying estimates"

    @computed_field
    def show_on_models_page(self) -> bool:
        return True

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        helper = SparklineEChartHelper(div_id=None,
                                       model_name='Static_Normals',
                                       v_unit='Mt_CO2e')
        helper.load_data()
        sector = PseudoSectors.Total_with_LULUCF
        rval = (helper.data_by_sector[sector]['lbound'],
                helper.data_by_sector[sector]['ubound'])
        return rval

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        helper = SparklineEChartHelper(div_id, v_unit, model_name=self.__class__.__name__)
        helper.load_data()
        helper.order_sectors()
        helper.add_total_cells()
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        return helper.make_echart()

    def GHGs_for_sector(self, sector):
        config = load_config(allow_version_mismatch=False)
        rval = []
        for ghg in GHG:
            if [str(sector), str(ghg)] in config['near_zero_sector_ghgs']:
                continue
            rval.append(ghg)
        return rval

    def sector_echart(self, sector, ghg, v_unit):
        helper = RegionalSparklineEChartHelper(
            sector=sector,
            ghg=ghg,
            div_id=f'{self.__class__.__name__}_regional_sparkline_echart_{ghg.value if ghg else "all"}',
            v_unit=v_unit)
        helper.load_data()
        helper.order_regions()
        helper.add_regional_cells()
        return helper.make_echart()

    def _challenge_score_PreNIR_2025_06(self):
        from . import nir_static_normals
        from . import model_db
        from scipy.special import logsumexp
        model_id = model_db.model_latest_version(
            family='StaticNormal',
            data_cutoff='2024-12-31',
            )['model_id']

        # product (log-domain sum) over components' predictions
        loglik_samples = 0
        for rd in model_db.components_by_model(model_id): # rd -> results/record dictionary
            if rd['component_type'] == 'Normal':
                loglik_samples += nir_static_normals.loglik_NIR_Normal(
                    rd['component_id'],
                    NIR_year=2025,
                    emission_year=2023)
            elif rd['component_type'] == 'BayesianNormal':
                loglik_samples += nir_static_normals.loglik_NIR_BayesianNormal(
                    rd['component_id'],
                    NIR_year=2025,
                    emission_year=2023)
            else:
                raise NotImplementedError(rd)

        # log-domain mean over samples
        rval = logsumexp(loglik_samples, b=1.0 / len(loglik_samples))
        return dict(total=float(rval))

    def challenge_scores(self, challenge_name):
        if challenge_name == 'PreNIR_2025_06':
            return self._challenge_score_PreNIR_2025_06()
        return dict(total=float('nan'))
