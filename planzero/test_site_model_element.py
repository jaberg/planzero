from collections.abc import Callable

from . import blog
from .enums import GHG, IPCC_Sector
from .prob import SiteModelElement


class SiteModelElementSmokeTester:
    assertIsInstance:Callable
    assertIsNotNone:Callable
    assertTrue:Callable
    assertEqual:Callable
    sme: SiteModelElement

    default_sector = IPCC_Sector.SCS__Public_Electricity_and_Heat
    default_ghg = GHG.CO2

    def test_description_html(self):
        rval = self.sme.description_html()
        self.assertIsInstance(rval, str)

    def test_short_description(self):
        rval = self.sme.short_description
        self.assertIsInstance(rval, str)

    def test_posts_developing_this_page(self):
        rval = self.sme.posts_developing_this_page
        self.assertIsInstance(rval, list)
        assert len(rval)
        for elem in rval:
            assert elem in blog.registry
