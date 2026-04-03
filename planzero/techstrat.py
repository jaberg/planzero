"""
A technology strategy is a possibility of using certain technologies together to
address a product expectation in the Canadian market.

For a example: meet the need for light-duty gasoline cars (directly connected
to an emissions sector, for which we have historical market data, scale
projections, etc.) with a new technology: battery-electric vehicles.

A technology strategy is different from a business, project or product: early
stage technology can be discussed in terms of technology strategy before
businesses have been created, and later-stage technology can be discussed in
terms of various business' projects or products that have aimed to realize
a technology strategy.

Technology Readiness Levels
===========================
Idea: the technology strategy can be described in words
Lab: the technology has been demonstrated in a lab setting (e.g. Interesting Engineering may write about it)
Investable: at least one investor has demonstrated commitment to develop the technology beyond the lab
Pilot: the technology has been used to power a usable product, early customers are trying it.
Scale: products and businesses are scaling to deliver the full potential of the technology to reduce emissions.
Mature: the products and businesses using this technology have met the full potential of the technology to reduce emissions.


See also https://climatewedges.com/index.html and paper in Science.

"""

import enum

class TechnologyReadinessLevel(str, enum.Enum):
    Idea = 'Idea'
    LabPrototype = 'Lab Prototype'
    Investable = 'Investable'
    Pilot = 'Pilot'
    Scale = 'Scale'
    Mature = 'Mature'

_classes = []

from .base import Project as DynamicElement

class TechStrat(DynamicElement):
    """Description in docstring will be picked up as long-form description"""
    # sectors affected by this technology strategy
    # Their analysis pages will feature links to this strategy
    ipcc_sectors: list[enums.IPCC_Sector]

    urls: dict[TechnologyReadinessLevel: list[str]]

    @property
    def TRL(self):
        for trl in reversed(TechnologyReadinessLevel):
            if self.urls[tr]:
                return trl
        else:
            return trl

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        _classes.append(cls)


class Light_Duty_BEV_250wh_per_kg(TechStrat):
    pass


class Light_Duty_BEV_500wh_per_kg(TechStrat):
    # Investable: https://interestingengineering.com/transportation/chinas-620-mile-solid-state-ev-battery
    pass


class Light_Duty_BEV_1000wh_per_kg(TechStrat):
    pass


class Heavy_Duty_BEV_250wh_per_kg(TechStrat):
    # TODO: how to communicate this:
    #       the battery truck cabs are in Scale
    #       but the batteries to compete with diesel are in LabPrototype
    #       ... so what's the technology?
    #       Maybe one technology is heavy duty vehicles based on 200 wh/kg?
    #       ... another technology is heavy duty vehicles based on e.g. 500 wh/kg?
    #       ... and anoth1000Wer technology is heavy duty vehicles based on e.g. 1000 wh/kg?
    pass

class Heavy_Duty_BEV_500wh_per_kg(TechStrat):
    # Investable: https://interestingengineering.com/transportation/chinas-620-mile-solid-state-ev-battery
    pass

class Heavy_Duty_BEV_1000wh_per_kg(TechStrat):
    """Heavy duty battery electric vehicles with 1000 wh/kg batteries can probably compete
    """
    # IndustryResearch: ['https://energyinnovation.org/wp-content/uploads/Delivering-Affordability-Emerging-Cost-Advantages-of-Battery-Electric-Heavy-Duty-Trucks.pdf']
    # LabPrototype: ['https://carnewschina.com/2025/12/16/china-battery-company-welion-achieves-824-wh-kg-energy-density-in-lab-targets-1000-wh-kg/']
    pass

class Methane_Pyrolysis_for_Public_Electricity(TechStrat): pass

class Airships_for_Timber_Removal(TechStrat): pass

class Biochar_as_Feed_Additive(TechStrat): pass

class Bovear_as_Feed_Additive(TechStrat): pass

class BatteryHarbourTugs(TechStrat): pass

class Nuclear_Public_Electricity(TechStrat): pass

class Solar_Public_Electricity(TechStrat): pass

class Wind_Public_Electricity(TechStrat):
    def __init__(self):
        super().__init__(
            urls={
                Scale: [
                    'https://www.bcbc.com/insight/ye64vvqox47trfwll459mre5y3cb18', # BC Business Council
                    'https://netzeroatlantic.ca/research-portal?type%5B0%5D=research_area%3A58',
                ],
            })

class Crop_Residue_Bio_Oil_for_CCS(TechStrat):
    def __init__(self):
        super().__init__(
        urls={
            Pilot: ['https://charmindustrial.com/'],
        })


class Zero_Bleed_Equipment_For_Natural_Gas(TechStrat):
    """Replace natural gas pneumatic equipment in the upstream oil and gas
    sector.

    This strategy is supported by Alberta Energy Regulator Directive 060 that
    pneumatic equipment must be replaced on failure, starting in 2023 (rather
    than maintained until
    mechanical end-of-life).
    """
    def __init__(self):
        super().__init__(
            urls={
                Scale: ['https://www.google.com/search?q=zero+bleed+pneumatic+controller&oq=zero+bleed+pneumatic+controller&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTINCAEQABiGAxiABBiKBTINCAIQABiGAxiABBiKBTIHCAMQABjvBTIKCAQQABiABBiiBDIHCAUQABjvBdIBCDY0MThqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:7e5ea2bf,vid:PqXb8fGuVT0,st:789'],
            })

class Vertical_Panel_Agrivoltaics(TechStrat):
    """Install vertical bifacial panels in fields to augment yield with
    electricity production.
    """
    def __init__(self):
        super().__init__(
            urls={
                Scale: [
                    'https://harvestingthesuntwice.org/', # h/t Paul Bird on LinkedIn
                ]
            )
