import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    idx_of_ghg,
    idx_of_pt,
    idx_of_sector,
    idx_of_year,
    ktCO2e_dense_w_nan,
    near_zero_sector_ghgs,
)
from .regional_total_emissions_element import PseudoRegion
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    years_key,
)
from .sector_total_emissions_element import PseudoSector
from .symmetric_blended_lognormal import (
    SymmetricBlendedLogNormal,
    kl_divergence_uniform_normal_mixture,
)


def abs_ktCO2e_weights(year:int) -> jnp.ndarray:
    # weights are abs( expected true value)
    # aka abs NIR2025 estimated national means
    abs_ktCO2e = np.zeros((len(IPCC_Sector), len(GHG), len(PT) - 1))
    arr_pt, arr_ca = ktCO2e_dense_w_nan()
    arr_idx_of_year = idx_of_year[year]
    abs_m_sgt = abs(arr_ca[:, :, arr_idx_of_year])
    abs_ktCO2e[:] = abs_m_sgt[:, :, None]

    abs_ktCO2e += np.where(np.isfinite(arr_pt[:, :, :, arr_idx_of_year]),
                           abs(arr_pt[:, :, :, arr_idx_of_year]),
                           0)

    # zero-out and ignore the tiny sector-gas combinations
    abs_ktCO2e[ abs_ktCO2e < 10 ] = 0
    return jnp.array(abs_ktCO2e / abs_ktCO2e.sum())


class SubtotalVarKey(GeneralKey, frozen=True):
    sector: IPCC_Sector | PseudoSector
    ghg: GHG|None
    region: PT | PseudoRegion
    nir_year: int
    months_ahead: int


class PreNIR_ModelElement(ModelElement):
    """
    Calculate a Pre-NIR-{year}-m{months_ahead} model metric. 
    """

    SubtotalVarKey = SubtotalVarKey

    nir_year:int # e.g. 2025
    months_ahead:int # e.g. 4
    nir_years_key:GeneralKey
    sector_ghg_nir_keys: dict[
            tuple[IPCC_Sector, GHG],
            dict[PT|PseudoRegion, GeneralKey]]
    sector_ghg_subtotal_keys: dict[
            IPCC_Sector,
            dict[GHG, GeneralKey|AnnualKey]]
    sector_region_ghg_subtotal_keys: dict[
            tuple[IPCC_Sector, PT],
            dict[GHG, GeneralKey|AnnualKey]]
    expect_all_sectors:bool

    def __init__(
            self,
            nir_year:int,
            months_ahead:int,
            nir_years_key:GeneralKey,
            sector_ghg_nir_keys: dict[
                tuple[IPCC_Sector, GHG],
                dict[PT|PseudoRegion, GeneralKey]],
            sector_ghg_subtotal_keys: dict[
                IPCC_Sector,
                dict[GHG, GeneralKey|AnnualKey]],
            sector_region_ghg_subtotal_keys: dict[
                tuple[IPCC_Sector, PT],
                dict[GHG, GeneralKey|AnnualKey]],
            expect_all_sectors:bool=True,
            ):
        super().__init__()
        self.nir_years_key = nir_years_key
        self.sector_ghg_nir_keys = sector_ghg_nir_keys
        self.sector_ghg_subtotal_keys = sector_ghg_subtotal_keys
        self.sector_region_ghg_subtotal_keys = sector_region_ghg_subtotal_keys
        self.expect_all_sectors = expect_all_sectors
        self.nir_year = nir_year
        self.months_ahead = months_ahead

    @staticmethod
    def element_identifier(nir_year, months_ahead) -> str:
        return f'PreNIR-{nir_year}-m{months_ahead}'

    @property
    def identifier(self) -> str:
        return self.element_identifier(self.nir_year, self.months_ahead)

    @property
    def prediction_year(self) -> int:
        if self.nir_year == 2025 and self.months_ahead == 4:
            return 2022
        else:
            raise NotImplementedError(self.nir_year, self.months_ahead)

    def _subtotal_var_key(self, sector, ghg, region):
        return SubtotalVarKey(
                elem_id=self.identifier,
                name='subtotal',
                sector=sector,
                ghg=ghg,
                region=region,
                nir_year=self.nir_year,
                months_ahead=self.months_ahead)

    @property
    def kl_keys(self) -> dict[
            tuple[IPCC_Sector,
                  GHG,
                  PT|PseudoRegion],
            SubtotalVarKey]:
        rval:dict[tuple[IPCC_Sector,
                        GHG,
                        PT|PseudoRegion],
                  SubtotalVarKey] = {}
        for sector in IPCC_Sector:
            for ghg in GHG:
                if (sector, ghg) in near_zero_sector_ghgs():
                    continue
                for pt in PT:
                    if pt == PT.XX:
                        continue
                    rval[sector, ghg, pt] = self._subtotal_var_key(sector, ghg, pt)
                rval[sector, ghg, PseudoRegion.NationalTotal] \
                        = self._subtotal_var_key(sector, ghg, PseudoRegion.NationalTotal)
        return rval

    @property
    def prenir_score_key(self) -> SubtotalVarKey:
        return self._subtotal_var_key(
                PseudoSector.Total_with_LULUCF,
                ghg=None,
                region=PseudoRegion.NationalTotal)

    @define(prenir_score_key, prior_shape=[], posterior_only=True)
    @define(kl_keys, prior_shape=[], posterior_only=True)
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Prior:
            # don't access the values to avoid sampling and slowing down MCMC
            pass
        elif ws.phase == Phase.Posterior:
            prenir_score = jnp.zeros(())
            weights = abs_ktCO2e_weights(self.prediction_year)
            nir_years = ws.val.general[self.nir_years_key]
            for ii, nir_year in enumerate(list(nir_years)):
                if nir_year == self.prediction_year:
                    nir_idx_of_prediction_year = ii
                    break
            else:
                raise NotImplementedError(self.prediction_year, nir_years)

            for key in self.kl_keys.values():
                assert isinstance(key.sector, IPCC_Sector)
                assert key.ghg is not None
                try:
                    nir_d = self.sector_ghg_nir_keys[key.sector, key.ghg]
                except KeyError:
                    if self.expect_all_sectors:
                        raise
                    else:
                        ws.val.general[key] = jnp.zeros(())
                        continue
                nir_dist_all_years_key = nir_d[key.region]
                nir_dist_all_years = ws.dist.general[nir_dist_all_years_key]
                if (isinstance(nir_dist_all_years, dists.ExpandedDistribution)
                    and isinstance(nir_dist_all_years.base_dist, SymmetricBlendedLogNormal)):
                    base_dist = nir_dist_all_years.base_dist
                    _, one = base_dist.shape()
                    assert one == 1, base_dist.shape()
                    # only integer getitem is implemented
                    nir_dist_1 = base_dist[nir_idx_of_prediction_year]
                    nir_dist = nir_dist_1[0]
                else:
                    raise NotImplementedError(nir_dist_all_years)

                if isinstance(key.region, PT):
                    weight = .5 * weights[idx_of_sector[key.sector],
                                          idx_of_ghg[key.ghg],
                                          idx_of_pt[key.region]]
                    try:
                        pred_key = self.sector_region_ghg_subtotal_keys[key.sector, key.region][key.ghg]
                    except KeyError as err:
                        err.add_note(f'key={key}')
                        raise
                elif key.region in PseudoRegion:
                    assert key.region == PseudoRegion.NationalTotal
                    weight = 0.5 * weights[idx_of_sector[key.sector],
                                          idx_of_ghg[key.ghg],
                                          :].sum()
                    try:
                        pred_key = self.sector_ghg_subtotal_keys[key.sector][key.ghg]
                    except KeyError as err:
                        err.add_note(f'key={key}')
                        raise
                else:
                    raise NotImplementedError(key.region)

                if isinstance(pred_key, GeneralKey):
                    pred_dist = ws.dist.general[pred_key]
                    prenir_score += self._helper(ws, key, weight, nir_dist, pred_dist)
                elif isinstance(pred_key, AnnualKey):
                    pred_dist = ws.dist.annual[pred_key]

                    # assigns ws.val.general[key] = kl
                    prenir_score += self._helper(ws, key, weight, nir_dist, pred_dist)
                else:
                    raise NotImplementedError(pred_key)

            ws.val.general[self.prenir_score_key] = prenir_score
        else:
            raise NotImplementedError(ws.phase)

    def _helper(self, ws, key, weight, nir_dist, pred_dist):
        model_years = ws.val.general[years_key]
        match_mask = model_years == self.prediction_year
        if jnp.any(match_mask):
            model_idx_of_prediction_year = jnp.argmax(match_mask)
        else:
            raise NotImplementedError(self.prediction_year)

        if isinstance(pred_dist, dists.Normal):
            pred_mu = pred_dist.loc
            pred_sigma_pt = pred_dist.scale
            mu_pred_year = pred_mu[model_idx_of_prediction_year]
            sigma_pred_year = pred_sigma_pt[model_idx_of_prediction_year]
            assert mu_pred_year.shape == (ws.n_mcmc,)
            assert sigma_pred_year.shape == (ws.n_mcmc,)
            assert nir_dist.shape() == ()
            kl = kl_divergence_uniform_normal_mixture(
                    nir_dist,
                    q_mu=mu_pred_year,
                    q_sigma=sigma_pred_year)
            assert jnp.isfinite(kl), (
                    key, mu_pred_year, sigma_pred_year)
            assert kl.shape == ()
            assert weight.shape == ()
            assert float(kl) >= 0
            assert float(weight) >= 0
            ws.val.general[key] = kl
            return kl * weight
        else:
            raise NotImplementedError(pred_dist)
