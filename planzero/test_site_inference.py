from collections.abc import Callable

from .enums import GHG, IPCC_Sector
from .prob import SiteInference
from .scenario_model import Computation


class SiteInferenceSmokeTester:
    assertIsInstance:Callable
    assertIsNotNone:Callable
    assertTrue:Callable
    assertEqual:Callable
    siteinf: SiteInference

    default_sector = IPCC_Sector.SCS__Public_Electricity_and_Heat
    default_ghg = GHG.CO2

    def test_registry_basename(self):
        result = self.siteinf.registry_basename()
        self.assertIsInstance(result, str)

    def test_scenario_model_registry_name(self):
        self.assertIsInstance(self.siteinf.scenario_model_registry_name, str)

    def test_scenario_model(self):
        sm = self.siteinf.scenario_model
        self.assertIsNotNone(sm)

    def test_comp(self):
        comp = self.siteinf.comp
        self.assertIsInstance(comp, Computation)

    def test_version_id(self):
        self.assertIsInstance(self.siteinf.version_hash(), str)

    def test_show_on_models_page(self):
        self.assertIsInstance(self.siteinf.show_on_models_page, bool)

    def test_short_description(self):
        self.assertIsInstance(self.siteinf.short_description, str)

    def test_challenge_results_url(self):
        self.assertEqual(
                self.siteinf.challenge_result_url('foobar'),
                '')

    def test_predicted_emissions_2050(self):
        self.assertIsInstance(
                self.siteinf.predicted_emissions_2050_MtCO2e_bounds_str,
                str)

    def test_registry_compute_model(self):
        comp = self.siteinf.registry_compute_model(compute_posterior=False)
        self.assertIsInstance(comp, Computation)

    def test_strategies(self):
        strategies = self.siteinf.strategies
        self.assertIsInstance(strategies, dict)

    def test_barriers(self):
        barriers = self.siteinf.barriers
        self.assertIsInstance(barriers, dict)

    def test_ghgs_for_sector(self):
        ghgs = self.siteinf.GHGs_for_sector(self.default_sector)
        self.assertIsInstance(ghgs, list)
        assert ghgs
        for ghg in ghgs:
            self.assertIsInstance(ghg, GHG)

    def test_sector_echart(self):
        result = self.siteinf.sector_echart(
                sector=self.default_sector,
                ghg=self.default_ghg,
                v_unit='Mt_CO2e')
        self.assertIsNotNone(result)

    def test_uncertain_sparkline_matrix_echart(self):
        result = self.siteinf.uncertain_sparkline_matrix_echart('div1', 'Mt_CO2e')
        self.assertIsNotNone(result)

    def test_ablations(self):
        for elem_id in self.siteinf.strategies:
            comp = self.siteinf.registry_compute_model(ablate_strategy_elem_id=elem_id)
            self.assertIsInstance(comp, Computation)

    def test_national_emissions_impact_echart(self):
        for elem_id in self.siteinf.strategies:
            echart = self.siteinf.national_emissions_impact_echart(
                    strategy_elem_id=elem_id,
                    div_id='div1',
                    v_unit='Mt_CO2e')
            self.assertIsNotNone(echart)

    def test_national_cost_impact_echart(self):
        for elem_id in self.siteinf.strategies:
            echart = self.siteinf.national_cost_impact_echart(
                    strategy_elem_id=elem_id,
                    div_id='div1',
                    v_unit='Billions CAD')
            self.assertIsNotNone(echart)
