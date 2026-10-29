import unittest

from .scaling_site import ScalingSiteInference
from .test_site_inference import SiteInferenceSmokeTester


class TestScalingSiteInference(unittest.TestCase, SiteInferenceSmokeTester):
    def setUp(self):
        self.siteinf = ScalingSiteInference()
