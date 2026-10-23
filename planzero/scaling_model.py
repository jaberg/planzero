import datetime

from . import bovaer_model
from .cattle_model import Cattle_StaticNormals_ModelElement
from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    near_zero_sector_ghgs,
)
from .nir2025_model import NIR2025_ModelElement
from .nir_model import (
    NIR_ScenarioModel,
    PreNIR_ModelElement,
    PseudoRegion,
    RegionalTotalEmissionsElement,
)
from .nir_static_normals_model import StaticNormals_ModelElement
from .scenario_model import (
    AnnualKey,
    GeneralKey,
)
from .sector_total_emissions_element import PseudoSector, SectorTotalEmissionsElement

n_regions = 13

def _add_emissions(self, sector, ghg, nir_elem, elem):
    self._sector_ghg_subtotal_keys[sector][ghg] = elem.ktCO2e_ca_key
    self._pbt_keys[sector, ghg] = elem.ktCO2e_ca_key

    nir_d = self._sector_ghg_nir_keys[sector, ghg] = {}
    nir_d[PseudoRegion.NationalTotal] = nir_elem.ca_key
    for pt in PT:
        if pt == PT.XX:
            continue
        dd = self._sector_region_ghg_subtotal_keys.setdefault((sector, pt), {})
        dd[ghg] = elem.ktCO2e_pt_keys[pt]
        nir_d[pt] = nir_elem.pt_keys[pt]


def _add_static_normals(self, sector, ghg, mcmc_group=None):
    mcmc_group = mcmc_group or f'{sector.value}_{ghg.value}'
    nir_elem = NIR2025_ModelElement(
            sector=sector,
            ghg=ghg,
            n_years=self.n_prior_years)
    self._nir_years_key = nir_elem.years_key
    self.add_element(nir_elem, mcmc_group=mcmc_group)
    if isinstance(self._sectors, list) and sector not in self._sectors:
        fit_to_data = False
    else:
        fit_to_data = True
    snorm_elem = StaticNormals_ModelElement(
                sector=sector,
                ghg=ghg,
                nir_scale_key=nir_elem.scale_key,
                nir_ca_key=nir_elem.ca_key,
                nir_pt_keys=nir_elem.pt_keys,
                fit_to_data=fit_to_data)
    self.add_element(snorm_elem, mcmc_group=mcmc_group)
    self._sector_ghg_subtotal_keys[sector][ghg] = snorm_elem.ktCO2e_ca_key
    self._pbt_keys[sector, ghg] = snorm_elem.ktCO2e_ca_key

    nir_d = self._sector_ghg_nir_keys[sector, ghg] = {}
    nir_d[PseudoRegion.NationalTotal] = nir_elem.ca_key
    for pt in PT:
        if pt == PT.XX:
            continue
        dd = self._sector_region_ghg_subtotal_keys.setdefault((sector, pt), {})
        dd[ghg] = snorm_elem.ktCO2e_pt_keys[pt]
        nir_d[pt] = nir_elem.pt_keys[pt]
    return nir_elem, snorm_elem


def _add_enteric_fermentation_and_other_product_mfg(self, mcmc_group=None):

    def no_mcmc(elem):
        return self.add_element(elem, mcmc_group=None)

    nir_elem_ef = self.add_element(
        NIR2025_ModelElement(
            sector=IPCC_Sector.Enteric_Fermentation,
            ghg=GHG.CH4,
            n_years=self.n_prior_years),
        mcmc_group=mcmc_group or f'{IPCC_Sector.Enteric_Fermentation.value}_{GHG.CH4.value}')

    self._nir_years_key = nir_elem_ef.years_key
    if isinstance(self._sectors, list) and IPCC_Sector.Enteric_Fermentation not in self._sectors:
        fit_to_data = False
    else:
        fit_to_data = True

    cattle_elem = self.add_element(
            Cattle_StaticNormals_ModelElement(
                nir_ca_key=nir_elem_ef.ca_key,
                nir_pt_keys=nir_elem_ef.pt_keys,
                model_headcounts=True,
                model_emission_factors=True,
                model_emissions=True,
                fit_to_data=fit_to_data,
                ),
            mcmc_group=mcmc_group or f'{IPCC_Sector.Enteric_Fermentation.value}_{GHG.CH4.value}')

    # Put this in a separate MCMC group for sampling efficiency, for as long
    # as the Bovaer model does not contemplate production that should already be
    # reflected in NIR data.
    nir_elem_opman, opman_elem = _add_static_normals(
            self,
            sector=IPCC_Sector.SCS__Manufacturing__Other,
            ghg=GHG.CO2,
            mcmc_group=mcmc_group)

    adoption_limit = no_mcmc(
            bovaer_model.Bovaer_Adoption_Limit())
    production_emissions = no_mcmc(
            bovaer_model.Bovaer_Production_Emissions(
                annual_headcounts_key=cattle_elem.annual_headcounts,
                annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction,
                baseline_ktCO2e_ca_key=opman_elem.ktCO2e_ca_key,
                baseline_ktCO2e_pt_keys=opman_elem.ktCO2e_pt_keys,
                ))
    monitoring = no_mcmc(
            bovaer_model.Bovaer_Monitoring(
                annual_headcounts_key=cattle_elem.annual_headcounts,
                annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction,
                ))
    farm_subsidy = no_mcmc(
            bovaer_model.Bovaer_Farm_Subsidy(
                def_bovine_fraction_on_bovaer=adoption_limit.bovine_fraction_on_bovaer,
                def_annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction,
                max_fraction=adoption_limit.max_possible_fraction_on_bovaer,
                annual_headcounts_key=cattle_elem.annual_headcounts,
                ))
    purchase_cost = no_mcmc(
            bovaer_model.Bovaer_Purchase_Cost(
                annual_headcounts_key=cattle_elem.annual_headcounts,
                annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction,
                ))
    no_mcmc(
            bovaer_model.Scale_Bovaer(
                def_bovaer_budget=farm_subsidy.bovaer_budget,
                ))

    _add_emissions(self, IPCC_Sector.Enteric_Fermentation, GHG.CH4,
                   nir_elem_ef, cattle_elem)
    _add_emissions(self, IPCC_Sector.SCS__Manufacturing__Other, GHG.CO2,
                   nir_elem_opman, production_emissions)

    #_add_subsidies(self, purchase_cost)
    #_add_subsidies(self, monitoring)
    #_add_subsidies(self, farm_subsidy)




class Scaling_ScenarioModel(NIR_ScenarioModel):
    def __init__(self,
                 data_cutoff:datetime.date,
                 last_forecast_year=2050,
                 sectors:type[IPCC_Sector]|list[IPCC_Sector]=IPCC_Sector,
                 num_warmup=1500,
                 num_samples=500,
                 ):
        """
        sectors: default to all sectors (site models require all sectors), but subsets can be used in testing
        """
        if data_cutoff == datetime.date(year=2024, month=12, day=31):
            last_observed_nir_year = 2022
        else:
            raise NotImplementedError(data_cutoff)
        assert 1990 <= last_observed_nir_year <= 2023
        super().__init__(
                first_year=1990,
                n_prior_years=last_observed_nir_year + 1 - 1990,
                n_posterior_years=last_forecast_year + 1 - 1990,
                num_warmup=num_warmup,
                num_samples=num_samples,
                )
        # num samples can be larger if scenario_model/computation.py
        # can free some memory as it goes... as of writing, it does not.
        self.data_cutoff = data_cutoff
        self._sector_ghg_subtotal_keys = {sector: {} for sector in IPCC_Sector}
        self._sector_region_ghg_subtotal_keys = {
                (sector, pt): {}
                for sector in IPCC_Sector for pt in PT if PT != PT.XX}
        self._sector_ghg_nir_keys:dict = {}
        self._pbt_keys:dict = {}
        self._nir_years_key = None
        self._sectors = sectors

        for sector in sectors:
            if sector in (IPCC_Sector.Enteric_Fermentation,
                          IPCC_Sector.SCS__Manufacturing__Other):
                pass
            else:
                for ghg in GHG:
                    if (sector, ghg) not in near_zero_sector_ghgs():
                        _add_static_normals(self, sector, ghg)
        _add_enteric_fermentation_and_other_product_mfg(self)

        self.add_element(
                SectorTotalEmissionsElement(
                    sector_ghg_subtotal_keys=self._sector_ghg_subtotal_keys,
                    expect_all_sectors=(sectors==IPCC_Sector),
                    ),
                mcmc_group=None)

        self.add_element(
                RegionalTotalEmissionsElement(
                    sector_region_ghg_subtotal_keys=self._sector_region_ghg_subtotal_keys,
                    expect_all_sectors=(sectors==IPCC_Sector),
                    ),
                mcmc_group=None)

        self.add_element_prenir_2025_m04(
                nir_years_key=self._nir_years_key,
                sector_ghg_nir_keys=self._sector_ghg_nir_keys,
                sector_ghg_subtotal_keys=self._sector_ghg_subtotal_keys,
                sector_region_ghg_subtotal_keys=self._sector_region_ghg_subtotal_keys,
                expect_all_sectors=(sectors==IPCC_Sector),
                )

        self.add_element_p_net_zero_2050(
                ktCO2e_normal_keys_by_sector_ghg=self._pbt_keys,
                )

    # NIR_ScenarioModel requirement
    def ktCO2e_var_key_sector_ghg_region(self, sector, ghg, pt) -> AnnualKey:
        elem_id = StaticNormals_ModelElement.element_identifier(sector, ghg)
        model_elem = self.model_elements[elem_id]
        if isinstance(model_elem, StaticNormals_ModelElement):
            if isinstance(pt, PT):
                return model_elem.ktCO2e_pt_keys[pt]
            elif isinstance(pt, PseudoRegion) and pt == PseudoRegion.NationalTotal:
                return model_elem.ktCO2e_ca_key
            else:
                raise NotImplementedError(pt)
        else:
            raise NotImplementedError()

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
