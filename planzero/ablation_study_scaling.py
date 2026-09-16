import datetime
from typing import ClassVar

import numpy as np
from pydantic import computed_field

from . import cattle, model_db, nir2025_site, prob, prob_bovaer
from .ablation import AblationStudy
from .annual_emission_results import AnnualEmissionResults
from .enums import GHG, Activity, IPCC_Sector, LULUCF_Sectors
from .nir_static_normals import (
    BNs_by_sector_ghg,
    model_id_from_data_cutoff,
    normals_by_sector_ghg,
    # weighted_KL_score,
)
from .sparkline_echart_helper import (
    PseudoRegion,
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

        self.years = np.arange(1990, 2050+1)

        #n_mu_timesteps_to_2022 = 31 # for NIR-2025
        #n_mu_timesteps_to_2050 = n_mu_timesteps_to_2022 + 27
        n_regions = 13

        n_samples = 500

        mean_with_lulucf = 0
        estimates_with_lulucf = np.zeros(
            (n_samples, len(self.years)))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros(
            (n_samples, len(self.years)))

        np_rng = np.random.default_rng(12345)

        static_normals_model_id = model_id_from_data_cutoff(
                datetime.date(year=2024, month=12, day=31))

        self.normals_by_sector_ghg = normals_by_sector_ghg(static_normals_model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(static_normals_model_id)

        for BN_d in self.BNs_by_sector_ghg.values():
            n_samples_ = BN_d['num_samples']
            assert n_samples_ == n_samples

        for sector in IPCC_Sector:

            estimated_sector_total_ca = np.zeros(
                (n_samples, len(self.years)))


            for ii, ghg in enumerate(GHG):
                if (sector, ghg) in self.normals_by_sector_ghg:
                    # This is assumed to be essentially 0
                    continue
                elif (sector, ghg) == (IPCC_Sector.Enteric_Fermentation, GHG.CH4):
                    from .prob_bovaer import batch_rollout_barriers
                    results = batch_rollout_barriers()
                    #estimated_sector_total_ca += results['ys']['enteric_fermentation_ktCO2e_ca_sample'].T  * self.v_unit_scale
                    estimated_sector_total_ca += results['ys']['enteric_fermentation_ktCO2e_ca_sample'].T  * self.v_unit_scale
                else:
                    BN_d = self.BNs_by_sector_ghg[sector, ghg]
                    grouped_samples = model_db.load_ndarray_group(
                            model_id=static_normals_model_id,
                            component_id=BN_d['component_id'],
                            group_id='grouped_samples')
                    n_chains, n_samples_, n_regions_ = grouped_samples['mu'].shape
                    assert n_chains == 1
                    assert n_regions == n_regions
                    assert n_samples_ == n_samples
                    samples_mu_ca = grouped_samples['mu'][0].sum(axis=1)
                    ca_sample = np_rng.standard_normal(n_samples)
                    ca_sample *= grouped_samples['sigma_ca'][0]
                    ca_sample += samples_mu_ca
                    ca_sample *= BN_d['scale'] * self.v_unit_scale

                    # broadcast out over timesteps
                    estimated_sector_total_ca += ca_sample[:, None]

            mean_sector_total = self.compute_stats_and_add_data_for_sector(
                sector,
                estimated_sector_total_ca)

            if sector not in LULUCF_Sectors:
                estimates_without_lulucf += estimated_sector_total_ca
                mean_without_lulucf += mean_sector_total
            estimates_with_lulucf += estimated_sector_total_ca
            mean_with_lulucf += mean_sector_total

        self.add_data_for_LULUCF_totals(
            estimates_with_lulucf,
            mean_with_lulucf,
            estimates_without_lulucf,
            mean_without_lulucf)

        # for drawing the reference values
        # this should be updated to e.g. 2026, 2027 etc. as available
        self.nir2025_sparkline_echart_helper = \
                nir2025_site.NIR2025_SparklineEChartHelper(
                        div_id='',
                        model_name='',
                        v_unit=self.v_unit)
        self.nir2025_sparkline_echart_helper.load_data()


class RegionalSparklineEChartHelper(RegionalSparklineEChartHelperBase):

    def load_data(self):
        static_normals_model_id = model_id_from_data_cutoff(
                datetime.date(year=2024, month=12, day=31))

        self.normals_by_sector_ghg = normals_by_sector_ghg(static_normals_model_id)
        self.BNs_by_sector_ghg = BNs_by_sector_ghg(static_normals_model_id)

        self.years = np.arange(1990, 2050+1)
        #n_mu_timesteps_to_2022 = 31 # for NIR-2025
        #n_mu_timesteps_to_2050 = n_mu_timesteps_to_2022 + 27
        n_regions = 13

        n_samples = 500

        np_rng = np.random.default_rng(12345)

        estimates_ghg_pt = np.zeros(
            (n_samples, len(GHG), len(self.years), n_regions))

        estimates_ghg_ca = np.zeros(
            (n_samples, len(GHG), len(self.years)))

        for ii, ghg in enumerate(GHG):
            if (self.sector, ghg) in self.normals_by_sector_ghg:
                # This is assumed to be essentially 0,
                pass
            elif (self.sector, ghg) == (IPCC_Sector.Enteric_Fermentation, GHG.CH4):
                from .prob_bovaer import batch_rollout_barriers
                results = batch_rollout_barriers()
                estimates_ghg_pt[:, ii, :, :] = (
                        results['ys']['enteric_fermentation_ktCO2e_pt_sample'].transpose((1, 0, 2))
                        * self.v_unit_scale)
                estimates_ghg_ca[:, ii, :] = (
                        results['ys']['enteric_fermentation_ktCO2e_ca_sample'].transpose()
                        * self.v_unit_scale)
            else:
                BN_d = self.BNs_by_sector_ghg[self.sector, ghg]
                grouped_samples = model_db.load_ndarray_group(
                        model_id=static_normals_model_id,
                        component_id=BN_d['component_id'],
                        group_id='grouped_samples')
                _, n_samples_, _ = grouped_samples['mu'].shape
                assert n_samples_ == n_samples
                samples_mu =  grouped_samples['mu'][0]
                pt_sample = np_rng.standard_normal((n_samples, 13))
                pt_sample *= grouped_samples['sigma_pt'][0]
                pt_sample += samples_mu
                pt_sample *= BN_d['scale'] * self.v_unit_scale
                estimates_ghg_pt[:, ii, :, :] = pt_sample[:, None, :]

                samples_mu_ca = grouped_samples['mu'][0].sum(axis=1)
                ca_sample = np_rng.standard_normal(n_samples)
                ca_sample *= grouped_samples['sigma_ca'][0]
                ca_sample += samples_mu_ca
                ca_sample *= BN_d['scale'] * self.v_unit_scale
                estimates_ghg_ca[:, ii, :] = ca_sample[:, None]


        estimates_pt = estimates_ghg_pt.sum(axis=1)
        estimates_ca = estimates_ghg_ca.sum(axis=1)

        self.add_data_from_estimates(
            estimates_pt=estimates_pt,
            estimates_ca=estimates_ca)


class EmissionImpactChartHelper(SparklineEChartHelperBase):
    """
    Show the emission impact of a strategy, in the context of an ablation study.
    """

    cells_include_actuals = False

    def load_data(self, aer:AnnualEmissionResults):
        self.years = aer.years

        n_samples = aer.n_samples

        mean_with_lulucf = 0
        estimates_with_lulucf = np.zeros(
            (n_samples, len(self.years)))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros(
            (n_samples, len(self.years)))

        from .results_ops import aer_sectors

        self.sectors = aer_sectors(aer)
        assert self.sectors
        # TODO: verify that the set of sectors calculated here
        # matches the set of sectors that was declared symbolically
        #affected_sectors = baseline.affected_sectors_by_strategy[strategy_name]

        self.n_non_lulucf_rows = 0
        while self.n_cols * self.n_non_lulucf_rows < len(self.sectors):
            self.n_non_lulucf_rows += 1
        self.n_total_rows = self.n_non_lulucf_rows

        for sector in self.sectors:
            if sector in LULUCF_Sectors:
                raise NotImplementedError()
            estimated_sector_total_ca = (
                    sum(aer[sector].values())
                    * self.v_unit_scale)
            mean_sector_total = self.compute_stats_and_add_data_for_sector(
                sector,
                estimated_sector_total_ca)

            if sector not in LULUCF_Sectors:
                estimates_without_lulucf += estimated_sector_total_ca
                mean_without_lulucf += mean_sector_total
            estimates_with_lulucf += estimated_sector_total_ca
            mean_with_lulucf += mean_sector_total

        self.add_data_for_LULUCF_totals(
            estimates_with_lulucf,
            mean_with_lulucf,
            estimates_without_lulucf,
            mean_without_lulucf)

    def order_sectors(self):

        # order sectors by decreasing last-year uncertainty
        non_lulucf_scores = [
            (-self.data_by_sector[sector]['ubound'], sector)
            for sector in self.sectors
            if sector not in LULUCF_Sectors]
        non_lulucf_scores.sort()
        self.sorted_non_lulucf = [sector for _, sector in non_lulucf_scores]
        self.sorted_lulucf = []


class ScalingSiteInference(prob.SiteInference):
    """Maximal deployment of existing products"""

    strategy_id: str|None
    data_cutoff:datetime.date = datetime.date(year=2024, month=12, day=31)

    @computed_field
    def ablation_study_id(self) -> str|None:
        return 'ScalingStudy'

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

    @computed_field
    def show_on_models_page(self) -> bool:
        return (self.strategy_id is None)

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        helper = SparklineEChartHelper(
                div_id,
                v_unit,
                model_name=self.name)
        helper.load_data()
        helper.order_sectors()
        helper.add_total_cells()
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        return helper.make_echart()

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

    def GHGs_for_sector(self, sector):
        static_normals_model_id = model_id_from_data_cutoff(
                datetime.date(year=2024, month=12, day=31))
        normals = normals_by_sector_ghg(static_normals_model_id)
        rval = []
        for ghg in GHG:
            if (sector, ghg) in normals:
                continue
            rval.append(ghg)
        return rval

    def impact_chart(self):
        return self.ablation_study.impact_chart(self.name)

    @property
    def affected_sectors_by_strategy(self) -> dict:
        baseline_rval = {
                'Scale_Bovaer': {IPCC_Sector.Enteric_Fermentation,
                                 IPCC_Sector.Other_Product_Manufacture_and_Use,},
                }
        if self.strategy_id is not None:
            del baseline_rval[self.strategy_id]
        return baseline_rval

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
    site_inference_names: dict[str|None, str]|None = None

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

    def emission_results(self, strategy_name):
        #baseline = prob.registry[self.site_inference_names[None]]
        #ablation = prob.registry[self.site_inference_names[strategy_name]]
        assert strategy_name == 'Scale_Bovaer'

        ktCO2e_sample = {}

        from .prob_bovaer import batch_rollout_barriers
        results = batch_rollout_barriers()

        # Enteric Fermentation
        sample_w_strategy = results['ys']['enteric_fermentation_ktCO2e_ca_sample'].T
        n_samples, n_years = sample_w_strategy.shape
        years = list(range(1990, 1990 + n_years))

        sample_wo_strategy = np.zeros_like(sample_w_strategy)
        sample_wo_strategy[:] = sample_w_strategy[:, 0][:, None]
        ktCO2e_sample[
                IPCC_Sector.Enteric_Fermentation,
                GHG.CH4,
                Activity.Farming_Cattle] = sample_w_strategy - sample_wo_strategy

        # Production Emissions delta
        ktCO2e_sample[
                IPCC_Sector.Other_Product_Manufacture_and_Use,
                GHG.CO2,
                Activity.Farming_Cattle] \
                        = results['ys']['bovaer_production_emissions_ktCO2e_ca_sample'].T


        return AnnualEmissionResults(
                ktCO2e_sample=ktCO2e_sample,
                years=years,
                n_samples=n_samples)

    @property
    def basename(self) -> str:
        return prob.registry[self.site_inference_names[None]].name

    @property
    def strategy_ids(self) -> list[str]:
        return ['Scale_Bovaer']

    def impact_chart(self, strategy_name:str):
        # TODO: move this to base class
        annual_emission_diffs = self.emission_results(strategy_name)
        helper = EmissionImpactChartHelper(
                div_id=f'emission_impact_echart_{self.basename}_{strategy_name}',
                v_unit='Mt_CO2e',
                model_name=self.basename)
        helper.load_data(annual_emission_diffs)
        helper.order_sectors()
        helper.add_total_cells(ymin_without_lulucf=None)
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        return helper.make_echart()

    def install_site_inferences(self):
        assert self.site_inference_names == None
        site_inference_names = {}

        site_inf = ScalingSiteInference(strategy_id=None)
        prob.registry[site_inf.name] = site_inf
        site_inference_names[None] = site_inf.name

        for strategy_id in self.strategy_ids:
            site_inf = ScalingSiteInference(strategy_id=strategy_id)
            prob.registry[site_inf.name] = site_inf
            site_inference_names[strategy_id] = site_inf.name

        self.site_inference_names = site_inference_names
