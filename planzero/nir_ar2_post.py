import datetime

from . import prob
from .blog import BlogPost, BlogStatus
from .enums import GHG, PT, IPCC_Sector, col_by_pt, col_ca
from .html import HTML_Matplotlib_Figure


class AR2(BlogPost):
    """
    This post introduces
    a new baseline probabilistic model based on two-step autoregressive
    processes (AR-2).
    Unlike the Static Normals model it does not assume year-to-year variations
    are all measurement error, it assumes that they are nearly accurate
    and that emissions change according to trends.
    This model estimates national, provincial and territorial emissions
    for every sector and greenhouse gas.
    Its near-term predictions are relatively precise, and longer-term
    predictions are less certain.
    """

    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 10, 28),
            title='Autoregressive Process Modelling of NIR-2025',
            url_filename="2026-05-26-probabilistic-modelling",
            author="James Bergstra",
            #tags={BlogTag.NIR_Modelling,
                 #},
            status=BlogStatus.Planned,
            published=False,
            )

    def generate_assets(self):

        for model_name, site_inference in prob.site_inferences.items():
            base = 'html/blog/2026-04-26-probabilistic-modelling'
            if model_name in ('Static_Normals', 'AR2'):
                site_inference.uncertain_sparkline_matrix_echart(
                    div_id=f"{model_name}_all_sectors",
                    v_unit="Mt_CO2e").save_as(
                        f'{base}-{model_name}-all_sectors.html')
                site_inference.sector_echart(
                    sector=IPCC_Sector.Harvested_Wood_Products,
                    ghg=GHG.CO2,
                    v_unit="Mt_CO2e").save_as(
                        f'{base}-{model_name}-HWP.html')
            else:
                continue

    def ar2_model(self, sector, ghg):
        from .nir_ar2 import NIR2025_AR2, load_config_samples
        _, grouped_samples = load_config_samples(sector, ghg)
        model = NIR2025_AR2(
            sector=sector,
            ghg=ghg,
            future_idx=NIR2025_AR2.idx_of_last_train_year(2022))
        model.grouped_samples = grouped_samples
        model.post_samples = {
            key: val.reshape(-1, *val.shape[2:])
            for key, val in grouped_samples.items()}
        return model

    def const_sector_ghg(self, ax, sector, ghg):
        from .nir_constant_predictor import NIR2025_Model
        from . import nir2025
        import numpy as np
        import jax.numpy as jnp
        from numpyro.diagnostics import hpdi

        model = NIR2025_Model.posterior_inference(
            sector=sector, ghg=ghg)
        predictions = model.predictions()

        mean_mu = jnp.mean(model.post_samples['mu'], axis=0)
        hpdi_mu_pt = hpdi(model.post_samples['mu'], 0.95)
        hpdi_mu_ca = hpdi(
            jnp.sum(model.post_samples['mu'], axis=1),
            0.95)

        # spread
        x = nir2025.nir2025_year_ints
        spread_ca = hpdi(predictions['obs_ca'], 0.95)
        ax.fill_between(
            x,
            np.ones(len(x)) * spread_ca[0] * model.scale,
            np.ones(len(x)) * spread_ca[1] * model.scale,
            alpha=0.1,
            interpolate=True,
            color=col_ca,
            )
        # mean
        ax.axhline(
            jnp.mean(predictions['obs_ca']) * model.scale,
            color=col_ca,
        )
        # data
        ax.scatter(
            nir2025.nir2025_year_ints,
            model.jnp_ca,
            color=col_ca,
        )

        for ii, pt in enumerate(PT):
            if pt == PT.XX:
                continue
            # spread
            spread_pt = hpdi(predictions['obs_pt'][ii], 0.95)
            ax.fill_between(
                x,
                np.ones(len(x)) * spread_pt[0] * model.scale,
                np.ones(len(x)) * spread_pt[1] * model.scale,
                alpha=0.1,
                interpolate=True,
                color=col_by_pt[pt],
                )
            # mean
            ax.axhline(
                jnp.mean(predictions['obs_pt'][ii]) * model.scale,
                color=col_ca,
            )
            # data
            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_pt[ii],
                color=col_by_pt[pt],
            )

    def foo(self,):
        assert 0
        sector_ghg_list = [
            (IPCC_Sector.SCS__Public_Electricity_and_Heat,
             GHG.CO2),
            (IPCC_Sector.Ammonia_Production,
             GHG.CO2),
            (IPCC_Sector.Cement_Production,
             GHG.CO2),
            (IPCC_Sector.SCS__Commercial_and_Institutional,
             GHG.CO2),
            (IPCC_Sector.Forest_Land,
             GHG.CO2),
            (IPCC_Sector.Harvested_Wood_Products,
             GHG.CO2),
            (IPCC_Sector.Lime_Production,
             GHG.CO2),
            (IPCC_Sector.Nitric_Acid_Production,
             GHG.N2O),
            (IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use,
             GHG.CO2),
            (IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use,
             GHG.N2O),
            (IPCC_Sector.Petrochemical_and_Carbon_Black_Production,
             GHG.CO2),
            (IPCC_Sector.SCS__Petroleum_Refining_Industries,
             GHG.CO2),
        ]

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(_, plt):
                n_cols = 3
                fig, axs = plt.subplots(4, n_cols, figsize=(10, 12))
                for row, axrow in enumerate(axs):
                    for col, ax in enumerate(axrow):
                        list_idx = col + row * n_cols
                        try:
                            sector, ghg = sector_ghg_list[list_idx]
                        except IndexError:
                            break
                        self.const_sector_ghg( ax, sector, ghg)
                        if col == 0:
                            ax.set_ylabel('Emissions (CO2e)')
                        ax.set_title(sector.value)
                plt.tight_layout()
        return RVAL()


    def figure_ar2_rhat(self):
        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2
        model = self.ar2_model(sector, ghg)
        grouped_samples = model.grouped_samples
        from numpyro.diagnostics import summary
        diagnostics = summary(grouped_samples, prob=0.90, group_by_chain=True)
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(_, plt):

                for param, stats in diagnostics.items():
                    plt.hist(stats['r_hat'].flatten(), alpha=.2, label=param)

                plt.legend(loc='upper right')
        return RVAL()

    def ar2_sector_ghg_pt(self, ax, sector, ghg, pt, list_idx, model):
        from . import nir2025
        import numpy as np
        import jax.numpy as jnp
        import jax.random as jrandom
        from numpyro.diagnostics import hpdi
        from . import nir_ar2

        rng_key = jrandom.key(1234)
        np_rng = np.random.default_rng(12345)
        rng_key, rng_key_ = jrandom.split(rng_key)

        n_mu_timesteps_to_2022 = 31 # for NIR-2025
        n_mu_timesteps_to_2050 = n_mu_timesteps_to_2022 + 27

        samples = nir_ar2.samples_with_extended_mu(
            model.post_samples,
            n_steps=n_mu_timesteps_to_2050 - n_mu_timesteps_to_2022,
            pt_rms=model.pt_rms,
            np_rng=np_rng)
        samples = nir_ar2.sample_past_given_mu(
            samples,
            relerr=model.relerr,
            pad_mu0_m1=True,
            key=rng_key_)

        rec_pt = samples['past-pt']
        rec_ca = samples['past-ca']
        rec_yrs = np.arange(1990, 2050)

        scale = model.scale / 1000 # convert to Mt

        if pt is None:
            # spread
            spread_ca = hpdi(rec_ca, 0.95)
            ax.fill_between(
                rec_yrs,
                spread_ca[0] * scale,
                spread_ca[1] * scale,
                alpha=0.1,
                interpolate=True,
                color=col_ca,
                label='$X$ 95% CI',
                )
            # mean
            ax.plot(
                rec_yrs,
                rec_ca.mean(axis=0) * scale,
                c=col_ca,
                label='$X$ mean',
                ls=":",
            )
            # data
            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_ca * scale / model.scale,
                color=col_ca,
                s=20,
                label='NIR-2025 data',
            )
            for ii, pt in enumerate(PT):
                if pt == PT.XX:
                    continue
                ax.scatter(
                    nir2025.nir2025_year_ints,
                    model.jnp_pt[ii] * scale / model.scale,
                    color=col_by_pt[pt],
                    alpha=.3,
                    s=20,
                )

        else:
            future_idx = len(model.scaled_ca)
            obs_valid = jnp.isfinite(model.scaled_pt[:, :future_idx])
            approx_obs = jnp.where(
                obs_valid,
                model.scaled_pt[:, :future_idx],
                jnp.nanmean(model.scaled_pt[:, :future_idx], axis=1, keepdims=True))

            mu = nir_ar2.complete_mu(samples)
            mu_spread = hpdi(mu, 0.95)
            mu_years = np.arange(mu.shape[1]) + 1990
            ax.fill_between(
                list(mu_years),
                mu_spread[0, :, list_idx] * scale,
                mu_spread[1, :, list_idx] * scale,
                alpha=0.2,
                interpolate=True,
                color=col_by_pt[pt],
                label=r'$\mu$ 95% CI',
                )

            spread_pt_ii = hpdi(rec_pt[:, :, list_idx], 0.95)
            ax.fill_between(
                rec_yrs,
                spread_pt_ii[0] * scale,
                spread_pt_ii[1] * scale,
                alpha=0.1,
                interpolate=True,
                color=col_by_pt[pt],
                label=r'$X$ 95% CI',
                )
            ax.plot(
                rec_yrs,
                np.mean(spread_pt_ii, axis=0) * scale,
                color=col_by_pt[pt],
                label=r'$X$ mean = $\mu$ mean',
                ls=':',
                )

            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_pt[list_idx] * scale / model.scale,
                color=col_by_pt[pt],
                label='NIR-2025 data',
                s=20,
            )

    def figure_ar2_hwp(self,):
        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2

        model = self.ar2_model(sector, ghg)

        import numpy as np

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(_, plt):
                n_cols = 2
                fig, axs = plt.subplots(7, n_cols, figsize=(9, 17))
                for row, axrow in enumerate(axs):
                    for col, ax in enumerate(axrow):
                        list_idx = col + row * n_cols
                        if list_idx == 0:
                            pt = None
                        else:
                            pt = list(PT)[list_idx - 1]
                        self.ar2_sector_ghg_pt(
                            ax, sector, ghg, pt, list_idx - 1,
                            model)
                        alpha_1 = np.mean(model.post_samples['alpha_1'][:, list_idx - 1])
                        alpha_2 = np.mean(model.post_samples['alpha_2'][:, list_idx - 1])
                        if pt is None:
                            ax.set_title("Canada (with regional subtotals)")
                        else:
                            ax.set_title(f'{pt.value} [$\\bar c_1={3 * alpha_1 - .5:.2f}$, $\\bar c_2={3 * alpha_2 - 1.5:.2f}$]')
                        ax.legend(loc='lower right')
                plt.tight_layout()
        return RVAL()

    def log_prob_2023_AR2(self):
        return float('nan')
        import numpy as np
        import planzero
        from planzero.nir_ar2 import NIR2025_AR2
        from planzero import nir2025

        IPCC_Sector = planzero.enums.IPCC_Sector
        GHG = planzero.enums.GHG

        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2
        model = self.ar2_model(sector, ghg)

        arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
        mu_ca = np.sum(model.post_samples['mu'], axis=2)

        arr_idx_of_2023 = arr_ca.shape[2] - 1
        mu_idx_of_2023 = arr_idx_of_2023 - 2 # 2 from it being AR2 model


        eval_mu = np.zeros((1000, 14))
        eval_mu[:, :13] = model.post_samples['mu'][:, mu_idx_of_2023]
        eval_mu[:, 13] = mu_ca[:, mu_idx_of_2023]

        eval_sigma = np.zeros(14)
        eval_sigma[:13] = model.noise_ca * model.pt_rms
        eval_sigma[13] = model.noise_ca

        eval_obs = np.zeros(14)
        eval_obs[:13] = arr_pt[
            nir2025.idx_of_sector[model.sector],
            nir2025.idx_of_ghg[model.ghg],
            :,
            arr_idx_of_2023]
        eval_obs[13] = arr_ca[
            nir2025.idx_of_sector[model.sector],
            nir2025.idx_of_ghg[model.ghg],
            arr_idx_of_2023]

        import numpyro.distributions as dist
        log_probs = dist.Normal(
            eval_mu * model.scale,
            eval_sigma * model.scale
        ).log_prob(
            eval_obs
        )
        logprob_X = np.sum(log_probs, axis=1)
        from scipy.special import logsumexp
        rval = logsumexp(logprob_X, b=1.0 / len(logprob_X))

        return rval


