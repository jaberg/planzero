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
from pydantic import BaseModel, computed_field
import yaml

from .my_functools import inference_cache
from . import nir2025
from .prob import SiteInference


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
    #numpyro.sample("forecast", dist.Normal(mu, sigma), sample_shape=(n_forecast,))


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


def plot_regression(x, y_mean, y_hpdi):
    # Sort values for plotting by x axis
    #idx = jnp.argsort(x)
    #marriage = x[idx]
    #mean = y_mean[idx]
    #hpdi = y_hpdi[:, idx]
    #divorce = dset.DivorceScaled.values[idx]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(x, uncenter(np.ones(len(x)) * y_mean))
    ax.plot(x, uncenter(centred_ca), "o")
    ax.fill_between(x, uncenter(np.ones(len(x)) * y_hpdi[0]), uncenter(np.ones(len(x)) * y_hpdi[1]), alpha=0.3, interpolate=True)
    return ax


def main():
    """Populate cache/inference/Static_Normals
    """
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    from .enums import IPCC_Sector, GHG

    config_data = {
        'near_zero_sector_ghgs': [],
        'predicted_emissions_2050_MtCO2e_bounds_ul': [0, 1000],
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
        sector_ghg_root = f'./cache/inference/Static_Normals/{str(ghg)}-{str(sector)}/'
        sector_ghg_config_path = f'{sector_ghg_root}/config.yaml'

        os.makedirs(sector_ghg_root, exist_ok=True)
        try:
            with open(sector_ghg_config_path, 'r') as prev_config_file:
                prev_sector_ghg_config = yaml.safe_load(prev_config_file) or {}
            prev_version = prev_sector_ghg_config.get('version', 0)
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
                f'{sector_ghg_root}/{key}.npy',
                mode='w+',
                dtype='float32', # save space
                shape=val.shape)
            fp[:] = val
            fp.flush()

        # Write the per-sector-gas summary
        sector_ghg_config = {
            'version': version,
            'scale': float(self.scale),
        }
        with open(sector_ghg_config_path, 'w') as file:
            yaml.safe_dump(sector_ghg_config, file, default_flow_style=False)



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

        def body_data():
            fontSize = 9
            rval = []
            rval.append(EChartMatrixBodyDataElem(
                coord=(0, 0),
                value='Total without LULUCF',
                label=dict(color='#999', fontSize=fontSize, position='insideTop'),
                ))
            list_of_sectors = [sector for sector in IPCC_Sector]
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

        grid_list = []
        xAxis_list = []
        yAxis_list = []
        series_list = []

        grid_list.append(
            EChartGrid(
                id='foo',
                coordinateSystem='matrix',
                #coord=(1, 0),
                #coord=(4, 3),
                coord=(6, 5),
                top=25,
                bottom=10,
                left='center',
                width='90%',
                containLabel=True,
                ))
        xAxis_list.append(
            EChartMatrixXAxis(
                type='category',
                id='foo',
                gridId='foo',
                scale=True,
                axisTick=dict(show=False),
                axisLabel=dict(show=False),
                axisLine=dict(show=False),
                splitLine=dict(show=False),
                ))
        yAxis_list.append(
            EChartMatrixYAxis(
                id='foo',
                gridId='foo',
                interval=1_000_000_000_000, # was: Number.MAX_SAFE_INTEGER
                scale=True,
                axisLabel=dict(showMaxLabel=True,fontSize=9),
                axisLine=dict(show=False),
                axisTick=dict(show=False),
                ))
        series_list.append(
            EChartSeriesBase(
                xAxisId='foo',
                yAxisId='foo',
                type='line',
                symbol='none',
                lineStyle=EChartLineStyle(lineWidth=1),
                data=[float(x) for x in np.random.randn(20)],
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
