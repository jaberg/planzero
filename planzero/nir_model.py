import datetime

import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dists

from .enums import GHG, PT, SPE, Activity, IPCC_Sector, LULUCF_Sectors
from .nir2025 import near_zero_sector_ghgs
from .prenir_element import PreNIR_ModelElement
from .regional_total_emissions_element import (
    PseudoRegion,
)
from .scenario_model import (
    AnnualKey,
    Computation,
    GeneralKey,
    ModelElement,
    Phase,
    ScenarioModel,
    VersionID,
    Workspace,
    access_annual,
    access_val,
    define_annual,
    years_key,
)
from .sector_total_emissions_element import (
    PseudoSector,
)


def custom_sum(args_iter):
    args = list(args_iter)
    if len(args) == 0:
        raise NotImplementedError()
    elif len(args) == 1:
        return args[0]
    else:
        return sum(args)


class NIR_ModelElement(ModelElement):

    @property
    def ktCO2e_emissions_ca(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], AnnualKey]:
        return {}

    @property
    def ktCO2e_emissions_pt(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], dict[PT, AnnualKey]]:
        return {}

    @property
    def CAD_cashflow(
            self,
            ) -> dict[tuple[SPE, Activity], AnnualKey]:
        return {}


class Sector_GHG_Observation(ModelElement):
    """
    Predict emissions from a single (sector, ghg) pair across provinces
    """

    sector:IPCC_Sector
    ghg:GHG
    ktCO2e_pt_contributors:dict[PT, list[AnnualKey]]
    ktCO2e_ca_contributors:dict[str, list[AnnualKey]]
    _nir_ca_key:GeneralKey
    _nir_pt_keys:dict[PT, GeneralKey]
    fit_to_data:bool

    def __init__(self,
                 sector:IPCC_Sector,
                 ghg:GHG,
                 nir_ca_key:GeneralKey,
                 nir_pt_keys:dict[PT, GeneralKey],
                 fit_to_data:bool=True,
                 ):
        super().__init__()
        self.sector = sector
        self.ghg = ghg
        self.ktCO2e_pt_contributors = {pt: [] for pt in PT if pt != PT.XX}
        self.ktCO2e_ca_contributors = {'ca': []}
        self._nir_ca_key = nir_ca_key
        self._nir_pt_keys = nir_pt_keys
        self.fit_to_data = fit_to_data

    @property
    def identifier(self) -> str:
        return f'Sector_GHG_Observation_{self.sector}_{self.ghg}'

    def add_contributor(self, pt_key_d:dict[PT, AnnualKey], ca_key:AnnualKey):
        for pt, key in pt_key_d.items():
            assert isinstance(key, AnnualKey)
            self.ktCO2e_pt_contributors[pt].append(key)
        self.ktCO2e_ca_contributors['ca'].append(ca_key)

    @property
    def ktCO2e_ca(self) -> AnnualKey:
        return self.annual_key(f'ktCO2e_ca_{self.sector}_{self.ghg}_{self.fit_to_data}')

    @property
    def ktCO2e_pt(self) -> dict[PT, AnnualKey]:
        return {pt: self.annual_key(f'ktCO2e_pt_{pt}_{self.sector}_{self.ghg}_{self.fit_to_data}')
                for pt in PT if pt != PT.XX}

    @property
    def nir_ca_key(self) -> GeneralKey:
        return self._nir_ca_key

    @property
    def nir_pt_keys(self) -> dict[PT, GeneralKey]:
        return self._nir_pt_keys

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @property
    def _contributors(self) -> dict[int, AnnualKey]:
        counter = 0
        rval = {}
        for keylist in self.ktCO2e_pt_contributors.values():
            for key in keylist:
                rval[counter] = key
                counter += 1
        for key in self.ktCO2e_ca_contributors['ca']:
            rval[counter] = key
            counter += 1
        return rval

    @property
    def _nir_years(self) -> int:
        return 2022 - 1990 + 1

    @access_val(nir_ca_key)
    @access_val(nir_pt_keys)
    @access_annual(_contributors, annual_shape=[])
    @define_annual(ktCO2e_ca, annual_shape=[])
    @define_annual(ktCO2e_pt, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):

        # nir key
        n_nir_years = ws.dist.general[self.nir_ca_key].shape()[0]

        n_regions = 13
        obs_weight = jnp.ones((ws.n_years,) + ws.mcmc_broadcast_shape)
        loc_sum = jnp.ones(())
        ssq_sum = jnp.ones(())
        for ii, (pt, keys) in enumerate(self.ktCO2e_pt_contributors.items()):

            if self.fit_to_data and ws.phase == Phase.Prior:
                ws.obs_weight.annual[self.ktCO2e_pt[pt]] = obs_weight
                assert ws.n_years >= n_nir_years, (ws.n_years, n_nir_years)
                if ws.n_years == n_nir_years:
                    ws.obs_dist.annual[self.ktCO2e_pt[pt]] = (
                            ws.dist.general[self.nir_pt_keys[pt]])
                else:
                    ws.obs_valid.annual[self.ktCO2e_pt[pt]] = (
                            jnp.array([True] * n_nir_years
                                      + [False] * (ws.n_years - n_nir_years)
                                      )[ws.mcmc_broadcast_index])
                    raise NotImplementedError('ws.obs_dist.annual[...] = ')

            if not keys:
                raise NotImplementedError()
            elif all(isinstance(ws.dist.annual[key], dists.Normal) for key in keys):
                loc = custom_sum(ws.dist.annual[key].loc for key in keys)
                ssq = custom_sum(ws.dist.annual[key].scale ** 2 for key in keys)
            else:
                raise NotImplementedError()

            loc_sum += loc
            ssq_sum += ssq

            ws.dist.annual[self.ktCO2e_pt[pt]] = dists.Normal(loc, jnp.sqrt(ssq))

        if self.fit_to_data and ws.phase == Phase.Prior:
            ws.obs_weight.annual[self.ktCO2e_ca] = obs_weight * n_regions
            if ws.n_years == n_nir_years:
                ws.obs_dist.annual[self.ktCO2e_ca] = ws.dist.general[self.nir_ca_key]
            else:
                raise NotImplementedError()

        ws.dist.annual[self.ktCO2e_ca] = dists.Normal(loc_sum, jnp.sqrt(ssq_sum))


class Sector_Sum_Over_GHG(ModelElement):
    """Compute a sum over the GHGs in a sector.
    This is used in reporting:
    * the regional totals support the panels on the top figure of the /model/foo/sector/bar page
    * the national total is used in the upper left of the top figure on /model/foo/sector/bar page
    * the national total is also used on the /model/foo page for this sector.
    """

    sector:IPCC_Sector
    use_distval_sum:bool
    ktCO2e_pt_contributors:dict[PT, list[AnnualKey]]
    ktCO2e_ca_contributors:dict[str, list[AnnualKey]]

    def __init__(self,
                 sector:IPCC_Sector,
                 use_dist_sum:bool=True):
        super().__init__()
        self.sector = sector
        self.use_dist_sum = use_dist_sum
        self.ktCO2e_pt_contributors = {pt: [] for pt in PT if pt != PT.XX}
        self.ktCO2e_ca_contributors = {'ca': []}

    @property
    def identifier(self) -> str:
        return f'Sector_Sum_Over_GHG_{self.sector}'

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @property
    def ktCO2e_ca(self) -> AnnualKey:
        return self.annual_key(f'ktCO2e_ca_{self.sector}_{self.use_dist_sum}')

    @property
    def ktCO2e_pt(self) -> dict[PT, AnnualKey]:
        return {pt: self.annual_key(f'ktCO2e_{pt}_{self.sector}_{self.use_dist_sum}')
                for pt in PT if pt != PT.XX}

    def add_contributor(self, pt_key_d:dict[PT, AnnualKey], ca_key:AnnualKey):
        for pt, key in pt_key_d.items():
            self.ktCO2e_pt_contributors[pt].append(key)
        self.ktCO2e_ca_contributors['ca'].append(ca_key)

    @property
    def _contributors(self) -> dict[int, AnnualKey]:
        counter = 0
        rval = {}
        for keylist in self.ktCO2e_pt_contributors.values():
            for key in keylist:
                rval[counter] = key
                counter += 1
        for key in self.ktCO2e_ca_contributors['ca']:
            rval[counter] = key
            counter += 1
        return rval

    @access_annual(_contributors, annual_shape=[])
    @define_annual(ktCO2e_ca, annual_shape=[], posterior_only=True)
    @define_annual(ktCO2e_pt, annual_shape=[], posterior_only=True)
    def model_element_postprocess(self, ws:Workspace):
        print('SSOG running', self.identifier)
        if ws.phase == Phase.Posterior:
            for ii, (pt, key) in enumerate(self.ktCO2e_pt.items()):
                ws.val.annual[key] = custom_sum(
                        ws.val.annual[ikey] for ikey in self.ktCO2e_pt_contributors[pt])

            ws.val.annual[self.ktCO2e_ca] = custom_sum(
                    ws.val.annual[key] for key in self.ktCO2e_pt.values())


class Sum_Over_Sectors(ModelElement):
    """Compute a sum over the sectors of the NIR.
    This is used in reporting:
    * credible interval for 2050 emissions
    * probability of net-zero by 2050
    * top left corner of main model page figure (without LULUCF)
    * top left corner of main model page figure, lower panel (with LULUCF)
    """

    _sector_totals:dict[IPCC_Sector, AnnualKey]
    include_LULUCF:bool
    use_dist_sum:bool

    def __init__(self,
                 include_LULUCF:bool,
                 use_dist_sum:bool=True,
                 ):
        super().__init__()
        self._sector_totals = {}
        self.include_LULUCF = include_LULUCF
        self.use_dist_sum = use_dist_sum

    @property
    def identifier(self) -> str:
        return f'Sum_Over_Sectors_{"w" if self.include_LULUCF else "wo"}_LULUCF'

    @property
    def version_id(self) -> VersionID:
        return (1,)

    def add_contributor(self, sector, key):
        assert sector not in self._sector_totals
        self._sector_totals[sector] = key
    
    @property
    def sector_totals(self) -> dict[IPCC_Sector, AnnualKey]:
        return self._sector_totals

    @property
    def ktCO2e_ca(self) -> AnnualKey:
        return self.annual_key(f'ktCO2e_ca_{self.include_LULUCF}_{self.use_dist_sum}')

    @access_annual(sector_totals)
    @define_annual(ktCO2e_ca, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Posterior:
            ws.val.annual[self.ktCO2e_ca] = sum(
                    ws.val.annual[key] for key in self.sector_totals.values())


class NIR_ScenarioModel(ScenarioModel):

    expect_all_sectors:bool
    sum_over_sectors:dict[bool, Sum_Over_Sectors]
    sector_sum_over_ghg:dict[IPCC_Sector, Sector_Sum_Over_GHG]
    sector_ghg_observations:dict[tuple[IPCC_Sector, GHG], Sector_GHG_Observation]

    # only information available at or before this date
    # may be used as conditioning observations in Phase.Prior
    data_cutoff:datetime.date

    def __init__(self, *,
                 last_forecast_year:int,
                 data_cutoff:datetime.date=datetime.date(year=2025, month=12, day=31),
                 expect_all_sectors:bool=True,
                 **kwargs):
        if data_cutoff >= datetime.date(year=2025, month=12, day=31):
            last_observed_nir_year = 2023
        elif data_cutoff >= datetime.date(year=2024, month=12, day=31):
            last_observed_nir_year = 2022
        else:
            raise NotImplementedError(data_cutoff)

        super().__init__(
                first_year=1990,
                n_prior_years=last_observed_nir_year + 1 - 1990,
                n_posterior_years=last_forecast_year + 1 - 1990,
                **kwargs)
        self.data_cutoff = data_cutoff
        self.expect_all_sectors = expect_all_sectors
        self.sum_over_sectors = {}
        self.sector_sum_over_ghg = {}
        self.sector_ghg_observations = {}

    def _latest_nir_model_element_before_cutoff(self, sector, ghg):
        assert isinstance(self.data_cutoff, datetime.date)
        if self.data_cutoff <= datetime.date(2024, 12, 31):
            from .nir2024_model import NIR2024_ModelElement
            nir_elem = NIR2024_ModelElement(
                    sector=sector,
                    ghg=ghg,
                    report_emissions=False)
        elif self.data_cutoff <= datetime.date(2025, 12, 31):
            from .nir2025_model import NIR2025_ModelElement
            nir_elem = NIR2025_ModelElement(
                    sector=sector,
                    ghg=ghg,
                    report_emissions=False)
        else:
            raise NotImplementedError()
        return nir_elem

    def _add_Sector_GHG_Observation(self, sector, ghg, mcmc_group, fit_to_data:bool, nir_elem):
        sgo = Sector_GHG_Observation(
                sector=sector,
                ghg=ghg,
                nir_ca_key=nir_elem.ca_key,
                nir_pt_keys=nir_elem.pt_keys,
                fit_to_data=fit_to_data,
                )
        self.add_element(sgo, mcmc_group=mcmc_group)
        return sgo

    def add_emissions_sums(self, fit_to_data:bool=True):
        self.sum_over_sectors[False] = Sum_Over_Sectors(include_LULUCF=False)
        self.sum_over_sectors[True] = Sum_Over_Sectors(include_LULUCF=True)
        model_elements_items = list(self.model_elements.items())
        for elem_id, elem in model_elements_items:
            if isinstance(elem, NIR_ModelElement):
                nme = elem
                mcmc_group = self.mcmc_group_by_elemid[elem_id]
                for (sector, ghg), pt_key_d in nme.ktCO2e_emissions_pt.items():

                    nir_elem = self._latest_nir_model_element_before_cutoff(sector, ghg)
                    if nir_elem.identifier not in self.model_elements:
                        self.add_element(nir_elem, mcmc_group=mcmc_group)
                    else:
                        nir_mcmc_group = self.mcmc_group_by_elemid[nir_elem.identifier]
                        if nir_mcmc_group != mcmc_group:
                            raise NotImplementedError()

                    if (sector, ghg) not in self.sector_ghg_observations:
                        sgo = self._add_Sector_GHG_Observation(sector, ghg, mcmc_group, fit_to_data, nir_elem)
                        self.sector_ghg_observations[sector, ghg] = sgo

                        if sector not in self.sector_sum_over_ghg:
                            ssog = Sector_Sum_Over_GHG(sector=sector)
                            self.add_element(ssog, mcmc_group=None)
                            self.sector_sum_over_ghg[sector] = ssog
                            self.sum_over_sectors[True].add_contributor(
                                    sector=sector,
                                    key=ssog.ktCO2e_ca)
                            if sector not in LULUCF_Sectors:
                                self.sum_over_sectors[False].add_contributor(
                                        sector=sector,
                                        key=ssog.ktCO2e_ca)
                        else:
                            ssog = self.sector_sum_over_ghg[sector]
                        ssog.add_contributor(sgo.ktCO2e_pt, sgo.ktCO2e_ca)
                    else:
                        sgo = self.sector_ghg_observations[sector, ghg]

                    sgo.add_contributor(
                            pt_key_d,
                            nme.ktCO2e_emissions_ca[sector, ghg])

                # end for (sector, ghg)
            else:
                # not isinstance(elem, NIR_ModelElement)
                pass
        # end for (elem_id, elem)

    def all_sector_echart_ktCO2e_sample(
            self,
            comp:Computation,
            ) -> dict[IPCC_Sector|PseudoSector, jnp.ndarray]:
        ktCO2e_sample = {}
        for sector in IPCC_Sector:
            ktCO2e_sample[sector] = comp.ndarray_d[
                    self.sector_sum_over_ghg[sector].ktCO2e_ca]
        ktCO2e_sample[PseudoSector.Total_with_LULUCF] = comp.ndarray_d[
                self.sum_over_sectors[True].ktCO2e_ca]
        ktCO2e_sample[PseudoSector.Total_without_LULUCF] = comp.ndarray_d[
                self.sum_over_sectors[False].ktCO2e_ca]
        return ktCO2e_sample

    def per_sector_echart_ktCO2e_sample(
            self,
            comp:Computation,
            sector:IPCC_Sector,
            ghg:GHG|None
            ) -> dict[PT|PseudoRegion, jnp.ndarray]:

        ktCO2e_sample:dict[PT|PseudoRegion, jnp.ndarray] = {}
        regions = [pt for pt in PT if pt != PT.XX] + [PseudoRegion.NationalTotal]
        if ghg is None:
            for region in regions:
                if isinstance(region, PT):
                    ktCO2e_sample[region] = comp.ndarray_d[
                            self.sector_sum_over_ghg[sector].ktCO2e_pt[region]]
                else:
                    ktCO2e_sample[region] = comp.ndarray_d[
                            self.sector_sum_over_ghg[sector].ktCO2e_ca]
        else:
            for region in regions:
                if isinstance(region, PT):
                    ktCO2e_sample[region] = comp.ndarray_d[
                            self.sector_ghg_observations[sector, ghg].ktCO2e_pt[region]]
                else:
                    ktCO2e_sample[region] = comp.ndarray_d[
                            self.sector_ghg_observations[sector, ghg].ktCO2e_ca]
        return ktCO2e_sample

    def predicted_emissions_2050_MtCO2e_bounds_95(
            self,
            comp:Computation,
            ) -> tuple[float, float]:
        ktCO2e_sample = comp.ndarray_d[
                self.sum_over_sectors[True].ktCO2e_ca]
        model_years = np.asarray(comp.ndarray_d[years_key])
        match_mask = (model_years == 2050)
        if np.any(match_mask):
            model_idx_of_prediction_year = np.argmax(match_mask)
        else:
            raise NotImplementedError('no prediction for 2050')
        ktCO2e_2050 = ktCO2e_sample[model_idx_of_prediction_year]
        n_mcmc, = ktCO2e_2050.shape
        assert n_mcmc > 100
        unit_conversion = 1 / 1000
        lower, upper = np.quantile(ktCO2e_2050, [0.025, 0.975]) * unit_conversion
        return lower, upper

    # PRE-NIR

    def prenir_2025_m04_total_key(self) -> GeneralKey:
        return PreNIR_ModelElement.SubtotalVarKey(
                elem_id=PreNIR_ModelElement.element_identifier(2025, 4),
                name="subtotal",
                sector=PseudoSector.Total_with_LULUCF,
                ghg=None,
                region=PseudoRegion.NationalTotal,
                nir_year=2025,
                months_ahead=4)

    def prenir_2025_m04_sector_ghg_key(self, sector:IPCC_Sector, ghg:GHG) -> GeneralKey:
        return PreNIR_ModelElement.SubtotalVarKey(
                elem_id=PreNIR_ModelElement.element_identifier(2025, 4),
                name="subtotal",
                sector=sector,
                ghg=ghg,
                region=PseudoRegion.NationalTotal,
                nir_year=2025,
                months_ahead=4)

    def prediction_scores_prenir_2025_m04(self, comp:Computation):
        scores_ca = {}
        near_zero = near_zero_sector_ghgs()
        for sector in IPCC_Sector:
            scores_ca[sector] = {}
            for ghg in GHG:
                if (sector, ghg) not in near_zero:
                    sector_ghg_key = self.prenir_2025_m04_sector_ghg_key(sector, ghg)
                    sector_ghg_val = np.asarray(comp.ndarray_d[sector_ghg_key])
                    # remove the mcmc dim
                    assert sector_ghg_val.shape == (1,)
                    scores_ca[sector][ghg] = float(sector_ghg_val[0])
        total_key = self.prenir_2025_m04_total_key()
        total_val = comp.ndarray_d[total_key]
        assert total_val.shape == (1,)
        rval = {
                'weighted_divergence': float(total_val[0]),
                'scores_ca': scores_ca,
                }
        return rval

    def _sector_ghg_subtotal_keys(self) -> dict[IPCC_Sector, dict[GHG, AnnualKey]]:
        rval = {sector: {} for sector in IPCC_Sector}
        for (sector, ghg), elem in self.sector_ghg_observations.items():
            rval[sector][ghg] = elem.ktCO2e_ca
        return rval

    def _sector_region_ghg_subtotal_keys(self):
        rval = {
                (sector, pt): {}
                for sector in IPCC_Sector for pt in PT if PT != PT.XX}
        real_PTs = [pt for pt in PT if pt != PT.XX]
        for (sector, ghg), elem in self.sector_ghg_observations.items():
            for pt in real_PTs:
                rval[sector, pt][ghg] = elem.ktCO2e_pt[pt]
        return rval

    def add_element_prenir_2025_m04(self):
        from .nir2025_model import NIR2025_ModelElement
        nir_years_key = None
        sector_ghg_nir_keys = {}
        real_PTs = [pt for pt in PT if pt != PT.XX]
        for (sector, ghg) in self.sector_ghg_observations:
            nir_elem = NIR2025_ModelElement(
                    sector=sector,
                    ghg=ghg,
                    report_emissions=False)
            nir_years_key = nir_elem.years_key
            self.add_element(nir_elem)

            sector_ghg_nir_keys[sector, ghg] = {}
            sector_ghg_nir_keys[sector, ghg][PseudoRegion.NationalTotal] = nir_elem.ca_key
            for pt in real_PTs:
                sector_ghg_nir_keys[sector, ghg][pt] = nir_elem.pt_keys[pt]

        assert nir_years_key is not None
        self.add_element(
                PreNIR_ModelElement(
                    nir_year=2025,
                    months_ahead=4,
                    nir_years_key=nir_years_key,
                    sector_ghg_nir_keys=sector_ghg_nir_keys,
                    sector_ghg_subtotal_keys=self._sector_ghg_subtotal_keys(),
                    sector_region_ghg_subtotal_keys=self._sector_region_ghg_subtotal_keys(),
                    expect_all_sectors=self.expect_all_sectors),
                mcmc_group=None)

    # P-Emissions-Below-Thresh
    def p_net_zero_2050(self, comp:Computation) -> float:
        # TODO: assert years end at 2050
        idx_2050 = -1
        total_ems = comp.get_ndarray(
                self.sum_over_sectors[False].ktCO2e_ca,
                sample_if_necessary=True)
        rval = float(jnp.mean(total_ems[idx_2050] < 0))
        return rval
