import datetime
from typing import ClassVar

import numpy as np
from pydantic import computed_field

from . import cattle, model_db, nir2025_site, prob
from .ablation import AblationStudy
from .enums import GHG, PT, IPCC_Sector, LULUCF_Sectors
from . import prob_bovaer

try:
    from .nir_static_normals import (
        BNs_by_sector_ghg,
        model_id_from_data_cutoff,
        normals_by_sector_ghg,
        # weighted_KL_score,
    )
except ImportError:
    pass
from .sparkline_echart_helper import (
        #PseudoRegion,
        PseudoSectors,
        RegionalSparklineEChartHelperBase,
        SparklineEChartHelperBase,
)

model_family = 'Scaling'
model_version = 1
model_version_description = "First draft"

def scaling_model_id(data_cutoff:datetime.date):
    static_normals_model_id = model_id_from_data_cutoff(data_cutoff)
    tuple_of_things = (
            model_family, model_version, data_cutoff,
            static_normals_model_id)
    model_hash = model_db.stable_hash(str(tuple_of_things))
    model_id = f'model_{model_hash}'
    return model_id


def touch_model(data_cutoff):
    with model_db.connect() as conn:
        cursor = conn.cursor()
        model_id = scaling_model_id(data_cutoff)

        try:
            model_db.by_id('Model', model_id=model_id)
        except model_db.NoRecord:
            model_db.insert_model(
                cursor=cursor,
                model_id=model_id,
                family=model_family,
                version=model_version,
                version_description=model_version_description,
                data_cutoff=data_cutoff,
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

    def load_data(self):

        static_normals_model_id = model_id_from_data_cutoff(
                datetime.date(year=2024, month=12, day=31))

        self.normals_by_sector_ghg = normals_by_sector_ghg(static_normals_model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(static_normals_model_id)


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

        rng = np.random.default_rng(seed=123)

        for sector in IPCC_Sector:
            sector_mean = 0

            estimates = rng.standard_normal((n_new_draws, n_samples, len(GHG)))

            for ii, ghg in enumerate(GHG):
                if (sector, ghg) in self.normals_by_sector_ghg:
                    estimates[:, :, ii] = 0
                else:
                    BN_d = self.BNs_by_sector_ghg[sector, ghg]
                    samples = model_db.load_ndarray_group(
                            model_id=static_normals_model_id,
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


class RegionalSparklineEChartHelper(RegionalSparklineEChartHelperBase):
    pass


class ScalingSiteInference(prob.SiteInference):
    """Maximal deployment of existing products"""

    strategy_id: str|None
    data_cutoff:datetime.date = datetime.date(year=2024, month=12, day=31)

    @computed_field
    def ablation_study_id(self) -> str|None:
        return 'Scaling'

    @property
    def name(self) -> str:
        if self.strategy_id is None:
            return 'ScalingStudy_All_Strategies'
        else:
            return f'ScalingStudy_minus_{self.strategy_id}'

    @computed_field
    def pretty_name(self) -> str:
        if self.strategy_id is None:
            return 'Scaling (All strategies)'
        else:
            return f'Scaling (minus {self.strategy_id})'

    @property
    def identifier(self) -> str:
        return scaling_model_id(self.data_cutoff)

    def main_inference_prep(self):
        #touch_model(self.data_cutoff)
        #touch_components(self.data_cutoff)
        pass


    @computed_field
    def show_on_models_page(self) -> bool:
        return (self.strategy_id is None)

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        helper = SparklineEChartHelper(
                div_id,
                v_unit,
                model_name=self.identifier)
        helper.load_data()
        helper.order_sectors()
        helper.add_total_cells()
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        return helper.make_echart()

    def main_model_id(self) -> int:
        # if model corresponds to a model in model_db, print model_id to
        # stdout and return 0
        print(self.identifier)
        return 0

    def main_inference_work(self):
        # called via e.g. __main__.py
        # post-condition: all inferences required for the model have been
        # computed and cached in local file system
        #
        # The required files may be cached, workers should check for completed
        # files and assume they are correct if they are present.
        prob_bovaer.cached_inference()


class ScalingStudy(AblationStudy):

    include_in_registry: ClassVar[bool] = True

    def barriers(self) -> list:
        return [
            cattle.Cattle_Population_AR(),
            cattle.Bovaer_Adoption_Limit(),
            cattle.Bovaer_Production_Emission_Factors(),
            cattle.Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
            cattle.Bovaer_Purchase_Cost(),
            cattle.Bovaer_Farm_Subsidy(),
            cattle.Bovaer_Monitoring(),
        ]

    @property
    def strategy_ids(self) -> list[str]:
        return ['Scale_Bovaer']

    def install_site_inferences(self):
        site_inf = ScalingSiteInference(strategy_id=None)
        prob.registry[site_inf.name] = site_inf

        for strategy_id in self.strategy_ids:
            site_inf = ScalingSiteInference(strategy_id=strategy_id)
