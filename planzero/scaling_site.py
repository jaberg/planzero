import datetime
import warnings
from typing import ClassVar

import jax.numpy as jnp
import numpy as np

from . import cattle, model_db, prob
from .ablation import AblationStudy
from .annual_emission_results import (
    AnnualEmissionResults,
    aer_sectors,
    total_prediction_CI,
)
from .annual_subsidy_results import NationalAnnualProgramBalances
from .challenge import NetZero_2050, PreNIR_2025_m04
from .enums import GHG, Activity, GovernmentProgram, IPCC_Sector, LULUCF_Sectors
from .model import compute_annual_emission_results
from .nir_static_normals import (
    BNs_by_sector_ghg,
    NIR_Sector_Static_Normal_Barrier,
    model_id_from_data_cutoff,
    normals_by_sector_ghg,
)
from .prob_bovaer import (
    Cattle_Population_Static_Normal,
    batch_rollout_barriers,
    cached_inference,
)
from .scaling_model import Scaling_ScenarioModel
from .sparkline_echart_helper import (
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
                    full_sample_ca = (
                            results['ys']['enteric_fermentation_ktCO2e_ca_sample']
                            * self.v_unit_scale)
                    n_years, n_per_post, _n_samples = full_sample_ca.shape
                    reshaped_sample_ca = full_sample_ca.reshape(
                            n_years, n_per_post * _n_samples).T
                    subsampled = np_rng.choice(reshaped_sample_ca,
                                               size=n_samples,
                                               replace=False)
                    estimated_sector_total_ca += subsampled
                else:
                    BN_d = self.BNs_by_sector_ghg[sector, ghg]
                    grouped_samples = model_db.load_ndarray_group(
                            model_id=static_normals_model_id,
                            component_id=BN_d['component_id'],
                            group_id='grouped_samples')
                    n_chains, n_samples_, n_regions_ = grouped_samples['mu'].shape
                    assert n_chains == 1
                    assert n_regions == n_regions_
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
                full_sample_pt = (
                        results['ys']['enteric_fermentation_ktCO2e_pt_sample']
                        * self.v_unit_scale)
                n_years, n_per_post, _n_samples, n_PTs = full_sample_pt.shape
                reshaped = full_sample_pt.reshape(
                        n_years, n_per_post * _n_samples, n_PTs)
                # put samples first for sub-sampling
                transposed = reshaped.transpose((1, 0, 2))
                subsampled = np_rng.choice(transposed,
                                           size=n_samples,
                                           replace=False)
                estimates_ghg_pt[:, ii, :, :] = subsampled

                full_sample_ca = (
                        results['ys']['enteric_fermentation_ktCO2e_ca_sample']
                        * self.v_unit_scale)
                n_years, n_per_post, _n_samples = full_sample_ca.shape
                reshaped_sample_ca = full_sample_ca.reshape(
                        n_years, n_per_post * _n_samples).T
                subsampled = np_rng.choice(reshaped_sample_ca,
                                           size=n_samples,
                                           replace=False)
                estimates_ghg_ca[:, ii, :] = subsampled

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


class ScalingSiteInference(prob.SiteInference):
    """Maximal deployment of available, modelled products,
    based on a baseline assumption of static emissions amounts."""

    include_in_registry:bool = False

    def __init__(self):
        data_cutoff = datetime.date(year=2024, month=12, day=31)
        # registry_name = (
        #         f'StaticNormals_data-cutoff'
        #         f'-{data_cutoff.year}'
        #         f'-{data_cutoff.month}'
        #         f'-{data_cutoff.day}'
        #         '_lfy-2050')
        registry_name = 'Scaling_2024_12_31'
        scenario_model = Scaling_ScenarioModel(
                data_cutoff=data_cutoff,
                last_forecast_year=2050,
                num_warmup=1500,
                num_samples=500,
                )
        super().__init__(
                scenario_model_registry_name=registry_name,
                scenario_model=scenario_model,
                )

    @property
    def pretty_name(self) -> str:
        return 'Scaling'

    def one_line_description(self):
        return "Hypothetical scaling of emission-reduction measures within an otherwise non-time-varying economy"

    @property
    def show_on_models_page(self) -> bool:
        return True

    def affected_sectors_by_strategy(self) -> dict:
        assert 'Scale_Bovaer' in self._strategies
        baseline_rval = {
                'Scale_Bovaer': {IPCC_Sector.Enteric_Fermentation,
                                 IPCC_Sector.Other_Product_Manufacture_and_Use,},
                }
        return baseline_rval

    @property
    def posts_developing_this_page(self) -> list[str]:
        return [
                'ProbabilisticBovaer',
                'StaticNormals',
                'ModellingBovaer',
                ]

    def challenge_result_url(self, challenge_name) -> str:
        if challenge_name in (
                'PreNIR_2025_m04',
                'NetZero_2050',
                ):
            return f'/models/prob/{self.url_name}/#{challenge_name}'
        else:
            return ''


class ScalingStudy(AblationStudy):

    include_in_registry: ClassVar[bool] = False
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
        n_years, n_per_post, n_samples = sample_w_strategy.shape
        years = list(range(1990, 1990 + n_years))

        # hack: estimate the emissions from not using the strategy
        # from the initial years of using the strategy, rather than the ablated
        # site inference.
        sample_wo_strategy = np.zeros_like(sample_w_strategy)
        sample_wo_strategy[:] = sample_w_strategy[0] # broadcast year 0
        ktCO2e_sample[
                IPCC_Sector.Enteric_Fermentation,
                GHG.CH4,
                Activity.Farming_Cattle] = (
                        sample_w_strategy
                        - sample_wo_strategy
                        ).reshape(n_years, -1)

        # Production Emissions delta
        sample_w_strategy_opmu = (
                np.zeros_like(sample_w_strategy)
                + results['ys']['bovaer_production_emissions_ktCO2e_ca_sample'][:, None, :])
        sample_wo_strategy_opmu = np.zeros_like(sample_w_strategy_opmu)

        ktCO2e_sample[
                IPCC_Sector.Other_Product_Manufacture_and_Use,
                GHG.CO2,
                Activity.Farming_Cattle] = (
                        sample_w_strategy_opmu
                        - sample_wo_strategy_opmu
                        ).reshape(n_years, -1)

        return AnnualEmissionResults(
                ktCO2e_sample=ktCO2e_sample,
                years=years,
                n_samples=n_samples * n_per_post)

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

        # upsample to match emission results
        CAD_sample = {
                key: (
                    np.zeros((n_years, 32, n_samples))
                    + (val.T)[:, None, :]
                    ).reshape((n_years, -1)).T
                for key, val in CAD_sample.items()
                }

        napb_no_net = NationalAnnualProgramBalances(
                CAD_sample=CAD_sample,
                years=years,
                n_samples=n_samples * 32)
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
                "Cattle_Enteric_Emission_Rates_NIR2025_Bovaer": cattle.Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(
                        calculate_KL_divergence_PreNIR_2025_m04=True,
                    ),
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
                        data_cutoff=datetime.date(year=2024, month=12, day=31),
                        calculate_KL_divergence_PreNIR_2025_m04=True,
                        )
                key = f'NIR_Sector_Static_Normal_{sector.value}'
                self._barriers[key] = barrier


def scaling_study_singleton():
    from . import ablation
    return ablation.registry['ScalingStudy']
