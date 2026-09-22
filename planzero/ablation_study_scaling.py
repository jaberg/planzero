import datetime
import warnings
from typing import ClassVar

import numpy as np
from pydantic import computed_field

from . import cattle, model_db, nir2025_site, prob
from .ablation import AblationStudy
from .annual_emission_results import (
        AnnualEmissionResults,
        aer_sectors,
        total_prediction_CI,
        )
from .annual_subsidy_results import NationalAnnualProgramBalances
from .enums import GHG, Activity, GovernmentProgram, IPCC_Sector, LULUCF_Sectors
from .model import compute_annual_emission_results
from .nir_static_normals import (
    BNs_by_sector_ghg,
    NIR_Sector_Static_Normal_Barrier,
    model_id_from_data_cutoff,
    normals_by_sector_ghg,
    # weighted_KL_score,
)
from .prob_bovaer import (
    Cattle_Population_Static_Normal,
    batch_rollout_barriers,
    cached_inference,
)
from .sparkline_echart_helper import (
    PseudoRegion,
    PseudoSectors,
    RegionalSparklineEChartHelperBase,
    SparklineEChartHelperBase,
    echart_from_napb,
)
from .strategies.strategy2 import Scale_Bovaer

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

    def load_data(self, batch_rollout_barriers_results):

        self.years = batch_rollout_barriers_results['years']

        # TODO: look these up from batch_rollout_barriers_results
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
                    results = batch_rollout_barriers_results
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

    def load_data(self, batch_rollout_barriers_results):
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
                results = batch_rollout_barriers_results
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
            (len(self.years), n_samples,))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros(
            (len(self.years), n_samples,))

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
                estimated_sector_total_ca.T)

            if sector not in LULUCF_Sectors:
                estimates_without_lulucf += estimated_sector_total_ca
                mean_without_lulucf += mean_sector_total
            estimates_with_lulucf += estimated_sector_total_ca
            mean_with_lulucf += mean_sector_total

        self.add_data_for_LULUCF_totals(
            estimates_with_lulucf.T,
            mean_with_lulucf,
            estimates_without_lulucf.T,
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
    """Maximal deployment of available, modelled products,
    based on a baseline assumption of static emissions amounts."""

    data_cutoff:datetime.date = datetime.date(year=2024, month=12, day=31)

    rollout_nbytes_budget:int = 50_000_000

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
            return 'Scaling'
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
        helper.load_data(
                batch_rollout_barriers_results=self.batch_rollout_barriers())
        helper.order_sectors()
        helper.add_total_cells()
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        rval = helper.make_echart()
        return rval

    def sector_echart(self, sector, ghg, v_unit):
        helper = RegionalSparklineEChartHelper(
            sector=sector,
            ghg=ghg,
            div_id=f'regional_sparkline_echart_{ghg.value if ghg else "all"}',
            v_unit=v_unit)
        helper.load_data(
                batch_rollout_barriers_results=self.batch_rollout_barriers())
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
        cached_inference()

    def batch_rollout_barriers(self):
        if not hasattr(self, '_rollout'):
            barriers = dict(self.ablation_study.barriers)
            strategies = {
                    name: obj
                    for name, obj in self.ablation_study.strategies.items()
                    if name != self.strategy_id}
            if self.strategy_id is not None:
                assert len(strategies) == len(self.ablation_study.strategies) - 1
            results = batch_rollout_barriers(
                    barriers=barriers,
                    strategies=strategies)
            nbytes = results.total_nbytes()
            if nbytes > self.rollout_nbytes_budget:
                warnings.warn(f'rollout exceeded bytes budget {nbytes} > {self.rollout_nbytes_budget}')
            self._rollout = results
        return self._rollout

    @property
    def posts_developing_this_page(self) -> list[str]:
        return [
                'ProbabilisticBovaer',
                'StaticNormals',
                'ModellingBovaer',
                ]

    def constants(self, elem_name):
        results = self.batch_rollout_barriers()
        var_names = results['constants'].outputs(elem_name)
        rval = {var_name: results['constants'][var_name]
                for var_name in var_names}
        return rval

    def full_time_series(self, elem_name):
        results = self.batch_rollout_barriers()
        var_names = results['ys_ic'].outputs(elem_name)
        rval = {var_name: results['ys'][var_name]
                for var_name in var_names}
        var_names_x = results['xs_ic'].outputs(elem_name)
        rval.update(
                {var_name: results['xs'][var_name]
                 for var_name in var_names_x})
        return rval

    def running_time_series(self, elem_name, include_rngs=False):
        results = self.batch_rollout_barriers()
        var_names = results['nc_ic'].outputs(elem_name)
        rval = {var_name: results['final_carry'][var_name]
                for var_name in var_names
                if include_rngs or 'key' not in str(results['final_carry'][var_name].dtype)
                }
        return rval

    @property
    def modelled_years(self):
        results = self.batch_rollout_barriers()
        return results['years']

    @property
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        aer = self.compute_annual_emission_results()
        low, high = total_prediction_CI(aer, year=2050)
        return float(low) / 1000, float(high) / 1000

    def compute_annual_emission_results(self) -> AnnualEmissionResults:
        return compute_annual_emission_results(
                strategies=self.strategies,
                barriers=self.barriers)


class ScalingStudy(AblationStudy):

    include_in_registry: ClassVar[bool] = True
    site_inference_names: dict[str|None, str]|None = None

    @property
    def _baseline_siteinf(self):
        baseline = prob.registry[self.site_inference_names[None]]
        return baseline

    @property
    def barriers(self) -> dict:
        return self._barriers

    @property
    def strategies(self) -> dict:
        return self._strategies

    def emission_results_delta(self, strategy_name:str):
        #ablation = prob.registry[self.site_inference_names[strategy_name]]
        assert strategy_name == 'Scale_Bovaer'

        ktCO2e_sample = {}

        results = self._baseline_siteinf.batch_rollout_barriers()

        # Enteric Fermentation
        sample_w_strategy = results['ys']['enteric_fermentation_ktCO2e_ca_sample']
        n_years, n_samples = sample_w_strategy.shape
        years = list(range(1990, 1990 + n_years))

        # hack: estimate the emissions from not using the strategy
        # from the initial years of using the strategy, rather than the ablated
        # site inference.
        sample_wo_strategy = np.zeros_like(sample_w_strategy)
        sample_wo_strategy[:] = sample_w_strategy[0] # broadcast year 0
        ktCO2e_sample[
                IPCC_Sector.Enteric_Fermentation,
                GHG.CH4,
                Activity.Farming_Cattle] = sample_w_strategy - sample_wo_strategy

        # Production Emissions delta
        ktCO2e_sample[
                IPCC_Sector.Other_Product_Manufacture_and_Use,
                GHG.CO2,
                Activity.Farming_Cattle] \
                        = results['ys']['bovaer_production_emissions_ktCO2e_ca_sample']

        return AnnualEmissionResults(
                ktCO2e_sample=ktCO2e_sample,
                years=years,
                n_samples=n_samples)

    def national_annual_program_balances(
            self,
            strategy_name:str,
            ) -> NationalAnnualProgramBalances:
        assert strategy_name == 'Scale_Bovaer'

        results = self._baseline_siteinf.batch_rollout_barriers()
        CAD_sample = {}

        sample_w_strategy = results['ys']['bovaer_farm_subsidy_ca'].T
        n_samples, n_years = sample_w_strategy.shape
        years = list(range(1990, 1990 + n_years))

        CAD_sample[GovernmentProgram.Bovine_Feed_Farm_Subsidy] \
                = -sample_w_strategy

        CAD_sample[GovernmentProgram.Income_Tax] = (
                   results['ys']['bovaer_farm_subsidy_ca_tax'].T
                   + results['ys']['bovaer_monitoring_admin_ca'].T
                   + results['ys']['bovaer_monitoring_onsite_ca'].T)

        CAD_sample[GovernmentProgram.Bovine_Feed_Cost] \
                = -results['ys']['bovaer_cost_ca'].T

        CAD_sample[GovernmentProgram.Bovine_Feed_Monitoring] \
                = -(results['ys']['bovaer_monitoring_admin_ca'].T
                    + results['ys']['bovaer_monitoring_onsite_ca'].T)

        napb_no_net = NationalAnnualProgramBalances(
                CAD_sample=CAD_sample,
                years=years,
                n_samples=n_samples)
        from .results_ops import napb_refresh_net
        napb = napb_refresh_net(napb_no_net)

        return napb

    @property
    def basename(self) -> str:
        return prob.registry[self.site_inference_names[None]].name

    @property
    def strategy_ids(self) -> list[str]:
        return ['Scale_Bovaer']

    def impact_chart(self, strategy_name:str):
        # TODO: move this to base class
        annual_emission_diffs = self.emission_results_delta(strategy_name)
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

    def government_impact_chart(self, strategy_name:str):
        napb = self.national_annual_program_balances(strategy_name)
        from .results_ops import napb_scale
        rval = echart_from_napb(
                napb=napb_scale(napb, 1 / 1_000_000_000),
                div_id=f'government_impact_echart_{self.basename}_{strategy_name}',
                model_name=self.basename)
        return rval

    def cost_per_tCO2e(self, strategy_name: str, q=(.025, .975)):
        napb = self.national_annual_program_balances(strategy_name)
        annual_emission_diffs = self.emission_results_delta(strategy_name)
        from . import results_ops
        rval_sample = results_ops.cost_per_tCO2e(napb=napb, aer=annual_emission_diffs)
        lower, upper = np.quantile(rval_sample, q=q)
        return lower, upper

    def model_post_init(self, context):
        # pydantic calls this after __init__() or model_construct()
        assert self.site_inference_names == None
        site_inference_names = {}

        site_inf = ScalingSiteInference(strategy_id=None)
        # as long as this class is used as a singleton, the assertion should pass
        assert site_inf.name not in prob.registry
        prob.registry[site_inf.name] = site_inf
        site_inference_names[None] = site_inf.name

        for strategy_id in self.strategy_ids:
            site_inf = ScalingSiteInference(strategy_id=strategy_id)
            assert site_inf.name not in prob.registry
            prob.registry[site_inf.name] = site_inf
            site_inference_names[strategy_id] = site_inf.name

        self.site_inference_names = site_inference_names

        self._barriers = {
                'Cattle_Population_Static_Normal': Cattle_Population_Static_Normal(),
                'Bovaer_Adoption_Limit': cattle.Bovaer_Adoption_Limit(),
                "Bovaer_Farm_Subsidy": cattle.Bovaer_Farm_Subsidy(),
                "Bovaer_Production_Emission_Factors": cattle.Bovaer_Production_Emission_Factors(),
                "Cattle_Enteric_Emission_Rates_NIR2025_Bovaer": cattle.Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
                "Bovaer_Purchase_Cost": cattle.Bovaer_Purchase_Cost(),
                "Bovaer_Monitoring": cattle.Bovaer_Monitoring(),
                }
        self._strategies = {
                "Scale_Bovaer": Scale_Bovaer(),
                }

        for sector in IPCC_Sector:
            if sector == IPCC_Sector.Enteric_Fermentation:
                # provided above by Bovaer_Production_Emission_Factors
                pass
            else:
                barrier = NIR_Sector_Static_Normal_Barrier(
                        sector=sector,
                        data_cutoff=datetime.date(year=2024, month=12, day=31))
                key = f'NIR_Sector_Static_Normal_{sector.value}'
                self._barriers[key] = barrier


def scaling_study_singleton():
    from . import ablation
    return ablation.registry['ScalingStudy']
