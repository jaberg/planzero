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
)
from .nir_static_normals_model import StaticNormals_ModelElement

n_regions = 13


def _add_static_normals(self, sector, ghg, mcmc_group=None):
    mcmc_group = mcmc_group or f'{sector.value}_{ghg.value}'
    nir_elem = NIR2025_ModelElement(
            sector=sector,
            ghg=ghg,
            report_emissions=False)
    self._nir_years_key = nir_elem.years_key
    self.add_element(nir_elem, mcmc_group=mcmc_group)
    snorm_elem = StaticNormals_ModelElement(sector=sector, ghg=ghg)
    self.add_element(snorm_elem, mcmc_group=mcmc_group)
    return nir_elem, snorm_elem


def _add_enteric_fermentation_and_other_product_mfg(self, mcmc_group=None):

    def no_mcmc(elem):
        return self.add_element(elem, mcmc_group=None)

    if isinstance(self._sectors, list) and IPCC_Sector.Enteric_Fermentation not in self._sectors:
        fit_to_data = False
    else:
        fit_to_data = True

    cattle_elem = self.add_element(
            Cattle_StaticNormals_ModelElement(
                model_headcounts=True,
                model_emission_factors=True,
                model_emissions=True,
                fit_to_data=fit_to_data,
                ),
            mcmc_group=mcmc_group or f'{IPCC_Sector.Enteric_Fermentation.value}_{GHG.CH4.value}')

    # Put this in a separate MCMC group for sampling efficiency, for as long
    # as the Bovaer model does not contemplate production that should already be
    # reflected in NIR data.
    _add_static_normals(
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
        super().__init__(
                data_cutoff=data_cutoff,
                last_forecast_year=last_forecast_year,
                num_warmup=num_warmup,
                num_samples=num_samples,
                expect_all_sectors=(sectors==IPCC_Sector),
                )
        # num samples can be larger if scenario_model/computation.py
        # can free some memory as it goes... as of writing, it does not.
        self.data_cutoff = data_cutoff
        self._sectors = sectors

        for sector in sectors:
            if sector in (IPCC_Sector.Enteric_Fermentation,
                          IPCC_Sector.SCS__Manufacturing__Other):
                pass
            else:
                for ghg in GHG:
                    if (sector, ghg) not in near_zero_sector_ghgs():
                        _add_static_normals(self, sector, ghg)
        _add_enteric_fermentation_and_other_product_mfg(self, mcmc_group='foo')

        self.add_emissions_sums()
        self.add_element_prenir_2025_m04()
