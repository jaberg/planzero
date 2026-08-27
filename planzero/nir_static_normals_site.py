import datetime
import enum

import numpy as np

from . import model_db, nir2025, nir2025_site
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
from .nir_static_normals import (
    BNs_by_sector_ghg,
    inference_work_loop,
    model_id_from_data_cutoff,
    normals_by_sector_ghg,
    touch_components,
    touch_model,
    weighted_KL_score,
)
from .prob import ClassVar, SiteInference, computed_field
from .sparkline_echart_helper import PseudoSectors, SparklineEChartHelperBase


class SparklineEChartHelper(SparklineEChartHelperBase):

    def add_static_data_for_sector(self, sector, sector_mean, lbound, ubound):
        assert sector not in self.data_by_sector
        assert ubound >= lbound
        self.data_by_sector[sector] = {
            'mean': sector_mean,
            'ubound': ubound,
            'lbound': lbound,
            'CI': ubound - lbound,
            'means': [sector_mean for yr in self.years],
            'ubounds': [ubound for yr in self.years],
            'lbounds': [lbound for yr in self.years],
            'CIs': [ubound - lbound for yr in self.years],
            'neg_shift': [min(ubound, 0) for yr in self.years],
            'neg_shade': [min(lbound, 0) - min(ubound, 0) for yr in self.years],
            'pos_shift': [max(lbound, 0) for yr in self.years],
            'pos_shade': [max(ubound, 0) - max(lbound, 0) for yr in self.years],
            }


    def load_data(self, model_id):

        self.normals_by_sector_ghg = normals_by_sector_ghg(model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(model_id)

        for BN_d in self.BNs_by_sector_ghg.values():
            n_samples = BN_d['num_samples']
            break
        else:
            assert 0, 'no BayesianNormal components found'
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
                if (sector, ghg) in self.normals_by_sector_ghg:
                    estimates[:, :, ii] = 0
                else:
                    BN_d = self.BNs_by_sector_ghg[sector, ghg]
                    samples = model_db.load_ndarray_group(
                            model_id=model_id,
                            component_id=BN_d['component_id'],
                            group_id='grouped_samples')
                    n_chains, n_samples_, n_regions = samples['mu'].shape
                    assert n_samples_ == n_samples
                    sector_mean_ghg = float(
                        samples['mu']
                        .reshape((n_chains * n_samples, n_regions))
                        .mean(axis=0) # across samples and chains
                        .sum(axis=0)) # over regions
                    sector_mean += sector_mean_ghg * BN_d['scale'] * self.v_unit_scale

                    estimates[:, :, ii] *= samples['sigma_ca']
                    estimates[:, :, ii] += samples['mu'].sum(axis=2)
                    estimates[:, :, ii] *= BN_d['scale'] * self.v_unit_scale

            sector_estimates = np.sum(estimates, axis=2)

            lbound, ubound = np.quantile(
                sector_estimates.flatten(),
                self.credibility_interval_95)

            self.add_static_data_for_sector(sector, sector_mean, lbound, ubound)

            if sector not in LULUCF_Sectors:
                estimates_without_lulucf += sector_estimates
                mean_without_lulucf += sector_mean
            estimates_with_lulucf += sector_estimates
            mean_with_lulucf += sector_mean

        lbound_with_lulucf, ubound_with_lulucf = np.quantile(
            estimates_with_lulucf.flatten(),
            self.credibility_interval_95)
        self.add_static_data_for_sector(
            PseudoSectors.Total_with_LULUCF,
            mean_with_lulucf,
            lbound_with_lulucf,
            ubound_with_lulucf)

        lbound_without_lulucf, ubound_without_lulucf = np.quantile(
            estimates_without_lulucf.flatten(),
            self.credibility_interval_95)
        self.add_static_data_for_sector(
            PseudoSectors.Total_without_LULUCF,
            mean_without_lulucf,
            lbound_without_lulucf,
            ubound_without_lulucf)

        # for drawing the reference values
        # this should be updated to e.g. 2026, 2027 etc. as available
        self.nir2025_sparkline_echart_helper = \
                nir2025_site.NIR2025_SparklineEChartHelper(
                        div_id='',
                        model_name='',
                        v_unit=self.v_unit)
        self.nir2025_sparkline_echart_helper.load_data()


class PseudoRegion(str, enum.Enum):
    NationalTotal = 'National Total'


class RegionalSparklineEChartHelper:

    """
    Build a minigrid for a single sector.
    It will have tiles for regions, arranged as 2 x 7.
    It may be a sum across GHGs or just one GHG
    """

    n_rows = 2
    n_cols = 7

    credibility_interval_95 = (.025, .975)

    def __init__(self, sector, ghg:GHG|None, div_id, v_unit, model_id):
        self.sector = sector
        self.ghg = ghg
        self.div_id = div_id
        self.grid_list = []
        self.xAxis_list = []
        self.yAxis_list = []
        self.series_list = []
        self.model_id = model_id

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
        self.data_by_region[region] = {
            'mean': mean,
            'ubound': ubound,
            'lbound': lbound,
            'CI': ubound - lbound,
            'means': [mean for yr in self.years],
            'ubounds': [ubound for yr in self.years],
            'lbounds': [lbound for yr in self.years],
            'CIs': [ubound - lbound for yr in self.years],
            'neg_shift': [min(ubound, 0) for yr in self.years],
            'neg_shade': [min(lbound, 0) - min(ubound, 0) for yr in self.years],
            'pos_shift': [max(lbound, 0) for yr in self.years],
            'pos_shade': [max(ubound, 0) - max(lbound, 0) for yr in self.years],
            }

    def load_data(self):
        self.normals_by_sector_ghg = normals_by_sector_ghg(self.model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(self.model_id)

        for BN_d in self.BNs_by_sector_ghg.values():
            n_samples = BN_d['num_samples']
            break
        else:
            assert 0, 'no BayesianNormal components found'
        n_new_draws = 125

        rng = np.random.default_rng(seed=123)
        estimates_ghg_pt = rng.standard_normal((n_new_draws, n_samples, len(GHG), 13))
        estimates_ghg_ca = rng.standard_normal((n_new_draws, n_samples, len(GHG)))

        for ii, ghg in enumerate(GHG):
            if ((self.sector, ghg) in self.normals_by_sector_ghg
                or (self.ghg is not None and self.ghg != ghg)):
                estimates_ghg_pt[:, :, ii] = 0
                estimates_ghg_ca[:, :, ii] = 0
            else:
                BN_d = self.BNs_by_sector_ghg[self.sector, ghg]
                samples = model_db.load_ndarray_group(
                        model_id=self.model_id,
                        component_id=BN_d['component_id'],
                        group_id='grouped_samples')
                _, n_samples_, _ = samples['mu'].shape
                assert n_samples_ == n_samples

                estimates_ghg_pt[:, :, ii] *= samples['sigma_pt']
                estimates_ghg_pt[:, :, ii] += samples['mu']
                estimates_ghg_pt[:, :, ii] *= BN_d['scale'] * self.v_unit_scale

                estimates_ghg_ca[:, :, ii] *= samples['sigma_ca']
                estimates_ghg_ca[:, :, ii] += samples['mu'].sum(axis=2)
                estimates_ghg_ca[:, :, ii] *= BN_d['scale'] * self.v_unit_scale

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
                        label={'color': '#999',
                               'fontSize': fontSize,
                               'position': 'insideTop'},
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
                    stack=f'stack_{region:s}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle={'opacity': .25},
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['neg_shade'])),
                    stack=f'stack_{region:s}'
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
                    stack=f'stack_{region:s}'
                    ))
            self.series_list.append(
                EChartSeriesBase(
                    name=f'{region} CI',
                    xAxisId=f'xAxis_{col}|{row}',
                    yAxisId=f'yAxis_{col}|{row}',
                    type='line',
                    symbol='none',
                    lineStyle=EChartLineStyle(opacity=0),
                    areaStyle={'opacity': .25},
                    itemStyle=EChartItemStyle(color=color),
                    data=list(zip(self.years, data['pos_shade'])),
                    stack=f'stack_{region:s}'
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

    @property
    def model_id(self):
        return model_id_from_data_cutoff(self.data_cutoff)

    def main_model_id(self) -> int:
        print(self.model_id)
        return 0

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
        helper = SparklineEChartHelper(
                div_id=None,
                model_name='Static_Normals',
                v_unit='Mt_CO2e')
        helper.load_data(self.model_id)
        sector = PseudoSectors.Total_with_LULUCF
        rval = (helper.data_by_sector[sector]['lbound'],
                helper.data_by_sector[sector]['ubound'])
        return rval

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        helper = SparklineEChartHelper(
                div_id,
                v_unit,
                model_name=self.__class__.__name__)
        helper.load_data(self.model_id)
        helper.order_sectors()
        helper.add_total_cells()
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        return helper.make_echart()

    def GHGs_for_sector(self, sector):
        normals = normals_by_sector_ghg(self.model_id)
        rval = []
        for ghg in GHG:
            if (sector, ghg) in normals:
                continue
            rval.append(ghg)
        return rval

    def sector_echart(self, sector, ghg, v_unit):
        helper = RegionalSparklineEChartHelper(
            sector=sector,
            ghg=ghg,
            div_id=f'{self.__class__.__name__}_regional_sparkline_echart_{ghg.value if ghg else "all"}',
            v_unit=v_unit,
            model_id=self.model_id)
        helper.load_data()
        helper.order_regions()
        helper.add_regional_cells()
        return helper.make_echart()

    def prediction_scores_prenir_2025_06(self):
        assert self.data_cutoff == datetime.date(year=2024, month=12, day=31)
        weighted_div, _, KLs = weighted_KL_score(
                year=2023, model_id=self.model_id)
        scores_ca = {}
        scores_pt = {}
        for ii, sector in enumerate(IPCC_Sector):
            scores_ca[sector] = dict(zip(GHG, KLs[ii, :, 13]))
            for jj, pt in enumerate(PT):
                if pt == PT.XX:
                    continue
                scores_pt[sector, pt] = dict(zip(GHG, KLs[ii, :, :13]))
        rval = {
                'weighted_divergence': weighted_div,
                'scores_ca': scores_ca,
                'scores_pt': scores_pt,
                }
        return rval

    def show_prediction_quality(self):
        assert self.data_cutoff == datetime.date(year=2024, month=12, day=31)
        return True
