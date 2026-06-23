"""
Example inference that's fast and represents a lower bound on accuracy.
"""
import os

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from pathlib import Path
from pydantic import BaseModel, computed_field
import yaml

from .my_functools import inference_cache
from . import nir2025
from .prob import SiteInference
from .enums import IPCC_Sector, GHG, PT


version = 0.1 # float


def constant_model(scaled_pt=None, scaled_ca=None):
    n_regions = 13
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand((n_regions,)))
    sigma_pt = numpyro.sample("sigma_pt",
                              dist.LogNormal(-1.0, 0.7).expand((n_regions,)))
    sigma_ca = numpyro.sample("sigma_ca",
                              dist.LogNormal(-1.0, 0.7))
    if scaled_pt is None:
        dist_pt = dist.Normal(mu, sigma_pt)
        obs_pt = None
    else:
        valid_pt = jnp.isfinite(scaled_pt)
        dist_pt = dist.Normal(mu, sigma_pt).mask(valid_pt)
        obs_pt = jnp.where(valid_pt, scaled_pt, 0)
    numpyro.sample("obs_pt", dist_pt, obs=obs_pt)

    numpyro.sample("obs_ca",
                   dist.Normal(jnp.sum(mu), sigma_ca),
                   obs=scaled_ca)


class NIR2025_Model(object):

    def __init__(self, sector, ghg, seed=0):
        self.sector = sector
        self.ghg = ghg
        self.seed = seed
        arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
        self.jnp_pt = jnp.array(arr_pt[nir2025.idx_of_sector[sector],
                                       nir2025.idx_of_ghg[ghg]],
                                dtype='float64')
        self.jnp_ca = jnp.array(arr_ca[nir2025.idx_of_sector[sector],
                                       nir2025.idx_of_ghg[ghg]],
                                dtype='float64')

        self.scale = max([
            np.sqrt(np.mean(self.jnp_ca ** 2)),
            np.sqrt(np.nanmean(np.nansum(self.jnp_pt ** 2, axis=0))),
            1.0,
        ])

        assert np.isfinite(self.scale)
        self.scaled_ca = jnp.array(self.jnp_ca / self.scale)
        self.scaled_pt = jnp.array(self.jnp_pt / self.scale)
        self.post_samples = None
        self.rng_key = jrandom.key(seed)

    @inference_cache()
    @staticmethod
    def posterior_inference(sector, ghg, seed=0,
                            num_warmup=250,
                            num_samples=1000,
                            version=0,
                            print_summary=False,
                            ret_mcmc=False):

        self = NIR2025_Model(sector, ghg, seed=seed)

        mcmc = MCMC(NUTS(constant_model),
                    num_warmup=250,
                    num_samples=num_samples)

        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        mcmc.run(rng_key_,
                 scaled_pt=self.scaled_pt.T,
                 scaled_ca=self.scaled_ca)
        if print_summary:
            mcmc.print_summary()
        self.post_samples = mcmc.get_samples()
        if ret_mcmc:
            return self, mcmc
        else:
            return self

    def predictions(self):
        predictive = Predictive(
            constant_model,
            self.post_samples,
            return_sites=['obs_pt', 'obs_ca'],
            )
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        predictions = predictive(rng_key_)
        return predictions


def model_root():
    return f'./cache/inference/Static_Normals'


def sector_ghg_root(sector, ghg):
    return f'./cache/inference/Static_Normals/{str(ghg)}-{str(sector)}'


def sector_ghg_config_path(sector, ghg):
    return f'{sector_ghg_root(sector, ghg)}/config.yaml'


class VersionMismatch(RuntimeError):
    pass

# TODO: refactor with nir_ar2

def load_config(allow_version_mismatch):
    with open(f'{model_root()}/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    versions_match = config.get('version', 0.1) == version
    if allow_version_mismatch or versions_match:
        return config
    raise VersionMismatch()


def sector_ghg_load_config(sector, ghg, allow_version_mismatch):
    with open(sector_ghg_config_path(sector, ghg), 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    versions_match = config.get('version', 0.1) == version
    if allow_version_mismatch or versions_match:
        return config
    raise VersionMismatch()


def load_config_samples(sector, ghg):
    config = sector_ghg_load_config(sector, ghg, allow_version_mismatch=False)
    rootdir = Path(sector_ghg_root(sector, ghg))

    # Dictionary to store the memory-mapped arrays
    mmap_dict = {}

    # Ensure the directory exists before iterating
    if rootdir.is_dir():
        # Loop over all files ending with .npy in the directory
        for file_path in rootdir.glob("*.npy"):
            # Extract the 'key' (the filename without the extension)
            key = file_path.stem
            try:
                # Load the file as a read-only memory-mapped array
                mmap_dict[key] = np.load(file_path, mmap_mode='r')
            except Exception as e:
                raise Exception(f"Error loading {file_path.name}: {e}") from e
    else:
        raise Exception(f"Directory not found: {rootdir}")

    return config, mmap_dict


def main():
    """Populate cache/inference/Static_Normals
    """
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    from .enums import IPCC_Sector, GHG

    config_data = {
        'near_zero_sector_ghgs': [],
        'version': version,
    }
    nontrivial = []
    for sector in IPCC_Sector:
        for ghg in GHG:
            jnp_pt = jnp.array(
                arr_pt[nir2025.idx_of_sector[sector],
                       nir2025.idx_of_ghg[ghg]],
                dtype='float64')
            jnp_ca = jnp.array(
                arr_ca[nir2025.idx_of_sector[sector],
                       nir2025.idx_of_ghg[ghg]],
                dtype='float64')
            if jnp.nansum(abs(jnp_ca)) <= 1: # 1kt, aka very small
                config_data['near_zero_sector_ghgs'].append(
                    [str(sector), str(ghg)])
            else:
                nontrivial.append((sector, ghg))

    # Write to the YAML file
    with open('./cache/inference/Static_Normals/config.yaml', 'w') as file:
        yaml.safe_dump(config_data, file, default_flow_style=False)

    for ii, (sector, ghg) in enumerate(nontrivial):
        root = sector_ghg_root(sector, ghg)
        config_path = sector_ghg_config_path(sector, ghg)

        os.makedirs(root, exist_ok=True)
        try:
            with open(config_path, 'r') as prev_config_file:
                prev_config = yaml.safe_load(prev_config_file) or {}
            prev_version = prev_config.get('version', 0)
            if prev_version != version:
                assert prev_version < version
                print ('updating old version', prev_version, 'to', version)
                raise RuntimeError('saved version is old')
            print(ii, '/', len(nontrivial), sector, ghg, 'done')
            continue
        except IOError:
            pass
        except RuntimeError:
            pass

        print(ii, '/', len(nontrivial), sector, ghg)
        self = NIR2025_Model(sector, ghg, seed=ii)

        mcmc = MCMC(NUTS(constant_model),
                    num_warmup=250,
                    num_samples=1000,
                    thinning=10)

        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        mcmc.run(rng_key_,
                 scaled_pt=self.scaled_pt.T,
                 scaled_ca=self.scaled_ca)
        grouped_samples = mcmc.get_samples(group_by_chain=True)
        for key, val in grouped_samples.items():
            fp = np.lib.format.open_memmap(
                f'{root}/{key}.npy',
                mode='w+',
                dtype='float32', # save space
                shape=val.shape)
            fp[:] = val
            fp.flush()

        # Write the per-sector-gas summary
        config = {
            'version': version,
            'scale': float(self.scale),
        }
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)


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
from .enums import IPCC_Sector, LULUCF_Sectors, col_by_pt, col_ca
import enum

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
        color = self.palette[(row * self.n_cols + col - 1) % len(self.palette)]

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
        self.grid_links.append(
            GridLinkElem(
                gridId=f'grid_{col}|{row}',
                url=f'/models/prob/{self.model_name}/sectors/{sector.value}',
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
                self.append_cell(row, col, sector, row_ymin * 0, row_ymax)

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
                self.append_cell(row, col, region, row_ymin * 0, row_ymax)

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


class Static_Normals(SiteInference):
    """Emissions per province and territory,
    and per greenhouse gas, are modelled as
    non-time-varying Normal distributions.
    National totals are modelled as the sums of provincial
    and territorial totals.
    This is a baseline model, not intended to be accurate.
    """

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
            div_id=f'regional_sparkline_echart_{ghg.value if ghg else "all"}',
            v_unit=v_unit)
        helper.load_data()
        helper.order_regions()
        helper.add_regional_cells()
        return helper.make_echart()


if __name__ == '__main__':
    import sys
    sys.exit(main())
