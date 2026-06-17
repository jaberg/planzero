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
from .enums import IPCC_Sector, GHG


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
        'predicted_emissions_2050_MtCO2e_bounds_ul': [0, 1000],
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



class Static_Normals(SiteInference):
    """Emissions per province and territory,
    and per greenhouse gas, are distributed according
    to non-time-varying Normal distributions.
    (This is a baseline model.)
    """

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        # Read the YAML file
        with open('./cache/inference/Static_Normals/config.yaml', 'r') as file:
            data = yaml.safe_load(file)
            return data['predicted_emissions_2050_MtCO2e_bounds_ul']

    def helper_sector_mean(self, config, sector, years):
        sector_mean = 0
        for ghg in GHG:
            if [str(sector), str(ghg)] in config['near_zero_sector_ghgs']:
                continue
            config_sg, samples = load_config_samples(sector, ghg)
            n_chains, n_samples, n_regions = samples['mu'].shape
            sector_mean_ghg = float(
                samples['mu']
                .reshape((n_chains * n_samples, n_regions))
                .mean(axis=0) # across samples and chains
                .sum(axis=0)) # over regions
            sector_mean += sector_mean_ghg * config_sg['scale']
            return [[yr, sector_mean] for yr in years]


    def uncertain_sparkline_matrix_echart(self, div_id):
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
            )
        from .enums import IPCC_Sector

        n_rows = 8
        n_cols = 9
        list_of_sectors = [sector for sector in IPCC_Sector]

        def body_data():
            fontSize = 9
            rval = []
            rval.append(EChartMatrixBodyDataElem(
                coord=[0, 0],
                value='Total without LULUCF',
                label=dict(color='#999', fontSize=fontSize, position='insideTop'),
                ))
            assert len(list_of_sectors) == 71
            for row in range(n_rows):
                for col in range(n_cols):
                    if row == col == 0:
                        continue
                    sector = list_of_sectors[row * n_cols + col - 1]
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
            return rval

        # just a few points since the prediction is constant, but
        # the data zoom should still work reasonably
        years = np.arange(1990, 2050+1, 10)
        config = load_config(allow_version_mismatch=False)

        grid_list = []
        xAxis_list = []
        yAxis_list = []
        series_list = []
        for row in range(n_rows):
            for col in range(n_cols):
                if row == col == 0:
                    continue
                sector = list_of_sectors[row * n_cols + col - 1]

                grid_list.append(
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
                xAxis_list.append(
                    EChartMatrixXAxis(
                        type='category',
                        id=f'xAxis_{col}|{row}',
                        gridId=f'grid_{col}|{row}',
                        scale=True,
                        axisTick=dict(show=False),
                        axisLabel=dict(show=False),
                        axisLine=dict(show=False),
                        splitLine=dict(show=False),
                        ))
                yAxis_list.append(
                    EChartMatrixYAxis(
                        id=f'yAxis_{col}|{row}',
                        gridId=f'grid_{col}|{row}',
                        interval=1_000_000_000_000, # was: Number.MAX_SAFE_INTEGER
                        scale=True,
                        axisLabel=dict(showMaxLabel=True,fontSize=9),
                        axisLine=dict(show=False),
                        axisTick=dict(show=False),
                        ))
                series_list.append(
                    EChartSeriesBase(
                        xAxisId=f'xAxis_{col}|{row}',
                        yAxisId=f'yAxis_{col}|{row}',
                        type='line',
                        symbol='none',
                        lineStyle=EChartLineStyle(width=2,lineWidth=1),
                        data=self.helper_sector_mean(config, sector, years),
                        ))

        rval = UncertainSparklineMatrixEChart(
            div_id=div_id,
            matrix=EChartMatrix(
                x=EChartMatrixXY(
                    length=9,
                    levelSize=40,
                    show=False,
                    ),
                y=EChartMatrixXY(
                    length=8,
                    levelSize=80,
                    show=False,
                    ),
                corner=EChartMatrixCorner( data=[], label={},),
                body=EChartMatrixBody(
                    data=body_data()),
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
            grid=grid_list,
            xAxis=xAxis_list,
            yAxis=yAxis_list,
            series=series_list,
            width='100%',
            height='900px',
            )
        return rval


if __name__ == '__main__':
    import sys
    sys.exit(main())
