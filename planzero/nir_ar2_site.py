from .nir_ar2 import *

from pydantic import computed_field

from .prob  import SiteInference
from . import nir_constant_predictor
from .enums import LULUCF_Sectors, PT
from .nir_constant_predictor import PseudoSectors, PseudoRegion


class SparklineEChartHelper(nir_constant_predictor.SparklineEChartHelper):
    # TODO: consider renaming this to reflect its ability to handle
    # time-varying bounds and means?

    def add_data_for_sector(self, sector, sector_means, lbounds, ubounds):
        assert sector not in self.data_by_sector
        assert np.all(ubounds >= lbounds)
        self.data_by_sector[sector] = dict(
            ubound=np.max(ubounds),
            lbound=np.min(lbounds),
            means=sector_means,
            ubounds=ubounds,
            lbounds=lbounds,
            CIs=ubounds - lbounds,
            neg_shift=np.minimum(ubounds, 0),
            neg_shade=np.minimum(lbounds, 0) - np.minimum(ubounds, 0),
            pos_shift=np.maximum(lbounds, 0),
            pos_shade=np.maximum(ubounds, 0) - np.maximum(lbounds, 0),
            )

    def compute_stats_and_add_data_for_sector(self, sector, sample):
        lbounds, ubounds = np.quantile(
            sample,
            q=self.credibility_interval_95,
            axis=0)

        mean_sector_total = np.mean(sample, axis=0)
        self.add_data_for_sector(
            sector,
            mean_sector_total,
            lbounds=lbounds,
            ubounds=ubounds)
        return mean_sector_total

    def add_data_for_LULUCF_totals(
        self,
        estimates_with_lulucf,
        mean_with_lulucf,
        estimates_without_lulucf,
        mean_without_lulucf,
        ):
        lbounds_with_lulucf, ubounds_with_lulucf = np.quantile(
            estimates_with_lulucf,
            q=self.credibility_interval_95,
            axis=0)

        self.add_data_for_sector(
            PseudoSectors.Total_with_LULUCF,
            mean_with_lulucf,
            lbounds=lbounds_with_lulucf,
            ubounds=ubounds_with_lulucf)

        lbounds_without_lulucf, ubounds_without_lulucf = np.quantile(
            estimates_without_lulucf,
            q=self.credibility_interval_95,
            axis=0)

        self.add_data_for_sector(
            PseudoSectors.Total_without_LULUCF,
            mean_without_lulucf,
            lbounds=lbounds_without_lulucf,
            ubounds=ubounds_without_lulucf)

    def load_data(self):
        self.years = np.arange(1990, 2050+1)
        self.config = load_config(allow_version_mismatch=False)

        n_samples = self.config['num_samples'] // self.config['thinning']
        n_mu_timesteps_to_2022 = 31 # for NIR-2025
        n_mu_timesteps_to_2050 = n_mu_timesteps_to_2022 + 27
        n_regions = 13

        mean_with_lulucf = 0
        estimates_with_lulucf = np.zeros(
            (n_samples, 2 + n_mu_timesteps_to_2050))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros(
            (n_samples, 2 + n_mu_timesteps_to_2050))

        rng_key = jrandom.key(1234)
        np_rng = np.random.default_rng(12345)

        for sector in IPCC_Sector:

            estimated_sector_total_ca = np.zeros(
                (n_samples, 2 + n_mu_timesteps_to_2050,))

            for ii, ghg in enumerate(GHG):
                if [str(sector), str(ghg)] in self.config['near_zero_sector_ghgs']:
                    continue
                else:
                    config_sg, grouped_samples = load_config_samples(sector, ghg)
                    n_chains, n_samples_, n_mu_timesteps_to_2022_, n_regions_ \
                            = grouped_samples['mu'].shape
                    try:
                        assert n_samples_ == n_samples
                        assert n_regions_ == n_regions
                        assert n_mu_timesteps_to_2022_ == n_mu_timesteps_to_2022
                    except AssertionError:
                        print(grouped_samples['mu'].shape)
                        raise

                    samples = {
                        key: val.reshape(-1, *val.shape[2:])
                        for key, val in grouped_samples.items()}

                    model = NIR2025_AR2(
                        sector,
                        ghg,
                        future_idx=NIR2025_AR2.idx_of_last_train_year(2022),
                    )

                    samples = samples_with_extended_mu(
                        samples,
                        n_steps=n_mu_timesteps_to_2050 - n_mu_timesteps_to_2022,
                        pt_rms=model.pt_rms,
                        np_rng=np_rng)

                    # TODO: samples_with_extended_mu

                    rng_key, rng_key_ = jrandom.split(rng_key)
                    samples = sample_past_given_mu(
                        samples,
                        relerr=model.relerr,
                        pad_mu0_m1=True,
                        key=rng_key_)

                    estimated_past_ca_ghg = (
                        samples['past-ca']
                        * config_sg['scale']
                        * self.v_unit_scale)

                    estimated_sector_total_ca += estimated_past_ca_ghg

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


class RegionalSparklineEChartHelper(nir_constant_predictor.RegionalSparklineEChartHelper):

    def add_data_for_region(self, region, means, lbounds, ubounds):
        assert region not in self.data_by_region
        assert np.all(ubounds >= lbounds)
        self.data_by_region[region] = dict(
            ubound=np.max(ubounds),
            lbound=np.min(lbounds),
            means=means,
            ubounds=ubounds,
            lbounds=lbounds,
            CIs=ubounds - lbounds,
            neg_shift=np.minimum(ubounds, 0),
            neg_shade=np.minimum(lbounds, 0) - np.minimum(ubounds, 0),
            pos_shift=np.maximum(lbounds, 0),
            pos_shade=np.maximum(ubounds, 0) - np.maximum(lbounds, 0),
            )

    def add_data_from_estimates(self, estimates_pt, estimates_ca):
        for ii, pt in enumerate(PT):
            if pt == PT.XX:
                continue
            pt_mean = np.mean(estimates_pt[:, :, ii], axis=0)
            pt_lbound, pt_ubound = np.quantile(
                estimates_pt[:, :, ii],
                self.credibility_interval_95,
                axis=0)
            self.add_data_for_region(pt, pt_mean, pt_lbound, pt_ubound)

        ca_mean = np.mean(estimates_ca, axis=0)
        ca_lbound, ca_ubound = np.quantile(
            estimates_ca,
            self.credibility_interval_95,
            axis=0)
        self.add_data_for_region(PseudoRegion.NationalTotal,
                                 ca_mean, ca_lbound, ca_ubound)

    def load_data(self):
        self.years = np.arange(1990, 2050+1)
        self.config = load_config(allow_version_mismatch=False)

        n_samples = self.config['num_samples'] // self.config['thinning']
        n_mu_timesteps_to_2022 = 31 # for NIR-2025
        n_mu_timesteps_to_2050 = n_mu_timesteps_to_2022 + 27
        n_regions = 13

        estimates_ghg_pt = np.zeros(
            (n_samples, len(GHG), 2 + n_mu_timesteps_to_2050, n_regions))

        estimates_ghg_ca = np.zeros(
            (n_samples, len(GHG), 2 + n_mu_timesteps_to_2050))

        rng_key = jrandom.key(1234)
        np_rng = np.random.default_rng(12345)

        for ii, ghg in enumerate(GHG):
            if [str(self.sector), str(ghg)] in self.config['near_zero_sector_ghgs']:
                estimates_ghg_pt[:, ii] = 0
                estimates_ghg_ca[:, ii] = 0
            elif self.ghg is not None and self.ghg != ghg:
                estimates_ghg_pt[:, ii] = 0
                estimates_ghg_ca[:, ii] = 0
            else:
                config_sg, grouped_samples = load_config_samples(self.sector, ghg)
                n_chains, n_samples_, n_mu_timesteps_to_2022_, n_regions_ \
                        = grouped_samples['mu'].shape
                try:
                    assert n_samples_ == n_samples
                    assert n_regions_ == n_regions
                    assert n_mu_timesteps_to_2022_ == n_mu_timesteps_to_2022
                except AssertionError:
                    print(grouped_samples['mu'].shape)
                    raise

                samples__ = {
                    key: val.reshape(-1, *val.shape[2:])
                    for key, val in grouped_samples.items()}

                model = NIR2025_AR2(
                    self.sector,
                    ghg,
                    future_idx=NIR2025_AR2.idx_of_last_train_year(2022),
                )

                samples_ = samples_with_extended_mu(
                    samples__,
                    n_steps=n_mu_timesteps_to_2050 - n_mu_timesteps_to_2022,
                    pt_rms=model.pt_rms,
                    np_rng=np_rng)

                rng_key, rng_key_ = jrandom.split(rng_key)
                samples = sample_past_given_mu(
                    samples_,
                    relerr=model.relerr,
                    pad_mu0_m1=True,
                    key=rng_key_)

                estimates_ghg_pt[:, ii] = (
                    samples['past-pt']
                    * config_sg['scale']
                    * self.v_unit_scale)

                estimates_ghg_ca[:, ii] = (
                    samples['past-ca']
                    * config_sg['scale']
                    * self.v_unit_scale)

        estimates_pt = estimates_ghg_pt.sum(axis=1)
        estimates_ca = estimates_ghg_ca.sum(axis=1)

        self.add_data_from_estimates(
            estimates_pt=estimates_pt,
            estimates_ca=estimates_ca)


class AR2(SiteInference):
    """Model emissions per province and territory,
    and per greenhouse gas, as evolving according
    to a 2-step autoregressive process (AR-2).
    """

    def one_line_description(self):
        return "Simple baseline model - 2-step autoregressive estimator"

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        helper = SparklineEChartHelper(div_id=None,
                                       model_name='AR2',
                                       v_unit='Mt_CO2e')
        helper.load_data()
        sector = PseudoSectors.Total_with_LULUCF
        rval = (helper.data_by_sector[sector]['lbounds'][-1],
                helper.data_by_sector[sector]['ubounds'][-1])
        return rval


    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        helper = SparklineEChartHelper(div_id, v_unit, model_name='AR2')
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
