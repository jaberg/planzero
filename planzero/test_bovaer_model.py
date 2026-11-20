import unittest

from .bovaer_model import (
    Bovaer_Adoption_Limit,
    Bovaer_Farm_Subsidy,
    Bovaer_Monitoring,
    Bovaer_Production_Emissions,
    Bovaer_Purchase_Cost,
    Scale_Bovaer,
)
from .scenario_model import AnnualKey, CarryKey
from .test_site_model_element import SiteModelElementSmokeTester


class Test_Bovaer_Adoption_Limit(unittest.TestCase, SiteModelElementSmokeTester):
    def setUp(self):
        self.sme = Bovaer_Adoption_Limit()


class Test_Production_Emissions(unittest.TestCase, SiteModelElementSmokeTester):
    def setUp(self):
        self.sme = Bovaer_Production_Emissions(
                AnnualKey(elem_id='dummy_elem_id', name='dummy_name'))


class Test_Monitoring(unittest.TestCase, SiteModelElementSmokeTester):
    def setUp(self):
        self.sme = Bovaer_Monitoring(
                annual_headcounts_key=AnnualKey(elem_id='dummy_elem_id', name='headcounts'),
                annual_bovaer_fraction=AnnualKey(elem_id='dummy_elem_id', name='fraction'),
                )


class Test_Farm_Subsidy(unittest.TestCase, SiteModelElementSmokeTester):
    def setUp(self):
        self.sme = Bovaer_Farm_Subsidy(
                def_bovine_fraction_on_bovaer=CarryKey(elem_id='dummy_elem_id', name='bovine_fraction_on_bovaer'),
                max_fraction=CarryKey(elem_id='dummy_elem_id', name='max_fraction'),
                annual_headcounts_key=AnnualKey(elem_id='dummy_elem_id', name='headcounts'),
                def_annual_bovaer_fraction=AnnualKey(elem_id='dummy_elem_id', name='fraction'),
                )


class Test_Purchase_Cost(unittest.TestCase, SiteModelElementSmokeTester):
    def setUp(self):
        self.sme = Bovaer_Purchase_Cost(
                annual_headcounts_key=AnnualKey(elem_id='dummy_elem_id', name='headcounts'),
                annual_bovaer_fraction=AnnualKey(elem_id='dummy_elem_id', name='fraction'),
                )


class Test_Scale_Bovaer(unittest.TestCase, SiteModelElementSmokeTester):
    def setUp(self):
        self.sme = Scale_Bovaer(
                def_bovaer_budget=AnnualKey(elem_id='dummy_elem_id', name='budget'),
                )
