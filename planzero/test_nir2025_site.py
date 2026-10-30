import unittest

from .nir2025_site import NIR2025
from .test_site_inference import SiteInferenceSmokeTester


class TestNIR2025SiteInference(unittest.TestCase, SiteInferenceSmokeTester):
    def setUp(self):
        self.siteinf = NIR2025()
