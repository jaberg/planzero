import datetime

from . import bovaer_model
from .cattle_model import Cattle_StaticNormals_ModelElement
from .enums import GHG, IPCC_Sector
from .nir2025 import (
    near_zero_sector_ghgs,
)
from .nir_model import (
    NIR_ScenarioModel,
)
from .nir_static_normals_model import StaticNormals_ModelElement


def _add_enteric_fermentation_and_other_product_mfg(self, mcmc_group:str):

    def add_element(elem):
        return self.add_element(elem, mcmc_group=mcmc_group)

    cattle_elem = add_element(
            Cattle_StaticNormals_ModelElement(
                model_headcounts=True,
                model_emission_factors=True,
                model_emissions=True,
            ))

    # TODO: assert that CO2 is only significant GHG from this sector
    add_element(
            StaticNormals_ModelElement(
                sector=IPCC_Sector.SCS__Manufacturing__Other,
                ghg=GHG.CO2))

    # the model is such that production emissions don't start until
    # after the years with data, so it should be fine to omit this element
    # from the MCMC, but in theory it should be included, and in practice
    # it shouldn't hurt.
    add_element(
            bovaer_model.Bovaer_Production_Emissions(
                annual_headcounts_key=cattle_elem.annual_headcounts,
                annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction))

    adoption_limit = add_element(
            bovaer_model.Bovaer_Adoption_Limit())

    farm_subsidy = add_element(
            bovaer_model.Bovaer_Farm_Subsidy(
                def_bovine_fraction_on_bovaer=adoption_limit.bovine_fraction_on_bovaer,
                def_annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction,
                max_fraction=adoption_limit.max_possible_fraction_on_bovaer,
                annual_headcounts_key=cattle_elem.annual_headcounts))

    # strategy
    add_element(
            bovaer_model.Scale_Bovaer(
                def_bovaer_budget=farm_subsidy.bovaer_budget))

    # cost tallying
    add_element(
            bovaer_model.Bovaer_Purchase_Cost(
                annual_headcounts_key=cattle_elem.annual_headcounts,
                annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction))
    add_element(
            bovaer_model.Bovaer_Monitoring(
                annual_headcounts_key=cattle_elem.annual_headcounts,
                annual_bovaer_fraction=cattle_elem.annual_bovaer_fraction))


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
            if sector == IPCC_Sector.Enteric_Fermentation:
                _add_enteric_fermentation_and_other_product_mfg(
                        self,
                        mcmc_group='enteric_and_other_product_mfg')
            elif sector == IPCC_Sector.SCS__Manufacturing__Other:
                for ghg in GHG:
                    if ghg == GHG.CO2:
                        # added by _add_enteric_fermentation_and_other_product_mfg
                        continue
                    if (sector, ghg) not in near_zero_sector_ghgs():
                        snorm_elem = StaticNormals_ModelElement(
                                sector=sector, ghg=ghg)
                        # match nir_static_normals_model.py
                        mcmc_group = f'{sector.value}_{ghg.value}'
                        self.add_element(snorm_elem,
                                         mcmc_group=mcmc_group)
                        self.mv.forgettable_dist_keys.update(
                                snorm_elem.ktCO2e_pt_keys.values())
                        self.mv.forgettable_dist_keys.add(
                                snorm_elem.ktCO2e_ca_key)
            else:
                for ghg in GHG:
                    if (sector, ghg) not in near_zero_sector_ghgs():
                        snorm_elem = StaticNormals_ModelElement(
                                sector=sector, ghg=ghg)
                        # match nir_static_normals_model.py
                        mcmc_group = f'{sector.value}_{ghg.value}'
                        self.add_element(snorm_elem,
                                         mcmc_group=mcmc_group)
                        self.mv.forgettable_dist_keys.update(
                                snorm_elem.ktCO2e_pt_keys.values())
                        self.mv.forgettable_dist_keys.add(
                                snorm_elem.ktCO2e_ca_key)

        self.add_emissions_sums()
        self.add_element_prenir_2025_m04()
