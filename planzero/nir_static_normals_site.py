import datetime

import numpy as np

from . import model_db
from .enums import GHG, PT, IPCC_Sector, LULUCF_Sectors
from .nir_static_normals import (
    BNs_by_sector_ghg,
    normals_by_sector_ghg,
    p_emissions_below_thresh_ex_LULUCF,
    weighted_KL_score,
)
from .nir_static_normals_model import StaticNormals_ScenarioModel
from .prob import SiteInference as SiteModel
from .scenario_model import ScenarioModel
from .sector_total_emissions_element import PseudoSector
from .sparkline_echart_helper import (
    PseudoRegion,
    RegionalSparklineEChartHelperBase,
    SparklineEChartHelperBase,
)


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

    def load_data(self, model_id, seed=123):

        self.normals_by_sector_ghg = normals_by_sector_ghg(model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(model_id)

        for BN_d in self.BNs_by_sector_ghg.values():
            n_samples = BN_d['num_samples']
            break
        else:
            assert 0, 'no BayesianNormal components found'
        n_new_draws = 32
        # empirically n_new_draws 1 with 500 samples gives different results from
        # n_new_draws 10, but above that it leads to fairly stable upper and lower
        # bounds on the national total

        mean_with_lulucf = 0
        estimates_with_lulucf = np.zeros((n_new_draws, n_samples))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros((n_new_draws, n_samples))

        rng = np.random.default_rng(seed=seed)
        for sector in IPCC_Sector:
            sector_mean = 0

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
            PseudoSector.Total_with_LULUCF,
            mean_with_lulucf,
            lbound_with_lulucf,
            ubound_with_lulucf)

        lbound_without_lulucf, ubound_without_lulucf = np.quantile(
            estimates_without_lulucf.flatten(),
            self.credibility_interval_95)
        self.add_static_data_for_sector(
            PseudoSector.Total_without_LULUCF,
            mean_without_lulucf,
            lbound_without_lulucf,
            ubound_without_lulucf)


class RegionalSparklineEChartHelper(RegionalSparklineEChartHelperBase):

    def add_static_data_for_region(self, region, mean, lbound, ubound):
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

    def load_data(self, model_id, seed=123):
        self.normals_by_sector_ghg = normals_by_sector_ghg(model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(model_id)

        for BN_d in self.BNs_by_sector_ghg.values():
            n_samples = BN_d['num_samples']
            break
        else:
            assert 0, 'no BayesianNormal components found'
        n_new_draws = 32

        rng = np.random.default_rng(seed=seed)
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
                        model_id=model_id,
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
            self.add_static_data_for_region(pt, pt_mean, pt_lbound, pt_ubound)

        ca_mean = np.mean(estimates_ca)
        ca_lbound, ca_ubound = np.quantile(
            estimates_ca.flatten(),
            self.credibility_interval_95)
        self.add_static_data_for_region(PseudoRegion.NationalTotal,
                                        ca_mean, ca_lbound, ca_ubound)


class Static_Normals_2024_12_31(SiteModel):
    """Emissions per province and territory,
    and per greenhouse gas, modelled as
    non-time-varying Normal distributions.
    National totals are modelled as the sums of provincial
    and territorial totals.
    This is a baseline model, not intended to be accurate.
    """

    include_in_registry:bool = True


    def __init__(self):
        data_cutoff = datetime.date(year=2024, month=12, day=31)
        # registry_name = (
        #         f'StaticNormals_data-cutoff'
        #         f'-{data_cutoff.year}'
        #         f'-{data_cutoff.month}'
        #         f'-{data_cutoff.day}'
        #         '_lfy-2050')
        registry_name = 'Static_Normals_2024_12_31'
        scenario_model = StaticNormals_ScenarioModel(
                data_cutoff=data_cutoff,
                last_forecast_year=2050,
                )
        super().__init__(
                scenario_model_registry_name=registry_name,
                scenario_model=scenario_model)

    def one_line_description(self) -> str:
        return "Very simple baseline model - non-time-varying estimates"

    @property
    def show_on_models_page(self) -> bool:
        return True

    def __TODO_prediction_scores_prenir_2025_m04(self):
        assert self.data_cutoff <= datetime.date(year=2024, month=12, day=31)
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

    def __TODO_challenge_netzero_2050(self):
        return p_emissions_below_thresh_ex_LULUCF(
                model_id=self.model_id,
                thresh_ktCO2e=0)

    def __TODO_show_prediction_quality(self):
        assert self.data_cutoff == datetime.date(year=2024, month=12, day=31)
        return True

    def __TODO_challenge_result_url(self, challenge_name) -> str:
        return ''
        if challenge_name in (
                'PreNIR_2025_m04',
                'NetZero_2050'):
            return f'/models/prob/{self.model_id}/#{challenge_name}'
        else:
            return ''

    @property
    def pretty_name(self) -> str:
        return 'Static Normals (data-cutoff: 2024-12-31)'

    @property
    def posts_developing_this_page(self) -> list[str]:
        return [
                'StaticNormals', # 2026-06-28
                'ProbabilisticNIR2025', # 2026-05-20
                ]
