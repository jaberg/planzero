import pint
from . import enums

ureg = pint.UnitRegistry()

ureg.define('CAD = [currency]')
ureg.define('USD = 1.35 CAD')

ureg.define('people = [human_population]')
ureg.define('cattle = [bovine_population]')

ureg.define('fraction = [] = frac')
ureg.define('ppm = 1e-6 fraction')
ureg.define('ppb = 1e-9 fraction')

for mass_substance in ['coal', 'coal_bit', 'coal_subbit',
                       'lignite',
                       'anthracite',
                       'petcoke',
                       'wood_od', # wood, oven-dried
                       'wood_mc25', # wood at 25% moisture content - the level for which Annex6 (table 6-1) lists combustion emission coefficients
                       'wood_mc50', # wood at 50% moisture content (e.g. green bark & branches)
                       'CO2', 'CO2e', 'CH4', 'N2O', 'HFC', 'PFC', 'SF6', 'NF3',
                       'carbon', # in e.g. wood
                       'steam',
                       'uranium',
                      ]:
    ureg.define(f"kg_{mass_substance} = [mass_{mass_substance}]")
    ureg.define(f"tonne_{mass_substance} = 1000 * kg_{mass_substance}")
    ureg.define(f"kilotonne_{mass_substance} = 1_000_000 * kg_{mass_substance} = kt_{mass_substance}")
    ureg.define(f"megatonne_{mass_substance} = 1_000_000_000 * kg_{mass_substance} = Mt_{mass_substance}")
    ureg.define(f"gigatonne_{mass_substance} = 1_000_000_000_000 * kg_{mass_substance} = Gt_{mass_substance}")

    ureg.define(f"pounds_{mass_substance} = 1 / 2.20462 * kg_{mass_substance}")

    ureg.define(f"g_{mass_substance} = 0.001 * kg_{mass_substance}")

# TODO: move to test file
assert ureg.Quantity("1 Mt_CO2").to('kg_CO2').magnitude == 1_000_000_000

for volume_substance in ['NG_mk', 'NG_nmk',
                         'crude',
                         'diesel',
                         'gasoline',
                         'kerosene', # aka jet fuel, stove oil
                         'HFO',  # Heavy Fuel Oil
                         'LFO',  # Light Fuel Oil
                         'LPGs', # Liquified petroleum gases
                         'NGLs', # Natural Gas Liquids
                         'RPPs', # Refined petroleum products
                         'stillgas',
                         'biogas',
                         'ovengas',
                         'petcoke',
                         'methane',
                         'propane',
                         'pulpingliquor',
                         'wood', 'hardwood', 'softwood',
                         'aviation_gasoline',
                         'aviation_turbo_fuel',
                        ]:
    ureg.define(f"m3_{volume_substance} = [volume_{volume_substance}]")
    ureg.define(f"l_{volume_substance} = .001 * m3_{volume_substance}")
    ureg.define(f"litres_{volume_substance} = .001 * m3_{volume_substance}")
    ureg.define(f"kilolitres_{volume_substance} = m3_{volume_substance}")
    ureg.define(f"megalitres_{volume_substance} = 1000 * m3_{volume_substance}")
    ureg.define(f"kilo_m3_{volume_substance} = 1000 * m3_{volume_substance}")
    ureg.define(f"mega_m3_{volume_substance} = 1000_000 * m3_{volume_substance}")

u = ureg
u.m3 = u.m ** 3

litres_by_fuel_type = {
    enums.FuelType.LightFuelOil: u.l_LFO,
    enums.FuelType.HeavyFuelOil: u.l_HFO,
    enums.FuelType.Diesel: u.l_diesel,
    enums.FuelType.Gasoline: u.l_gasoline,
    enums.FuelType.Kerosene: u.l_kerosene,
    enums.FuelType.PetCoke: u.l_petcoke,
    enums.FuelType.StillGas: u.l_stillgas,
}

m3_by_fuel_type = {
    enums.FuelType.NaturalGasMkt: u.m3_NG_mk,
    enums.FuelType.NaturalGasNonMkt: u.m3_NG_nmk,
    enums.FuelType.StillGas: u.m3_stillgas,
}

kg_by_ghg = {
    enums.GHG.CO2: u.kg_CO2,
    enums.GHG.CH4: u.kg_CH4,
    enums.GHG.N2O: u.kg_N2O,
    enums.GHG.HFCs: u.kg_HFC,
    enums.GHG.PFCs: u.kg_PFC,
    enums.GHG.SF6: u.kg_SF6,
    enums.GHG.NF3: u.kg_NF3,
}

kt_by_ghg = {
    enums.GHG.CO2: u.kt_CO2,
    enums.GHG.CH4: u.kt_CH4,
    enums.GHG.N2O: u.kt_N2O,
    enums.GHG.HFCs: u.kt_HFC,
    enums.GHG.PFCs: u.kt_PFC,
    enums.GHG.SF6: u.kt_SF6,
    enums.GHG.NF3: u.kt_NF3,
}

tonne_by_coal_type = {
    enums.CoalType.CanadianBituminous: u.tonne_coal_bit,
    enums.CoalType.CanadianSubbituminous: u.tonne_coal_subbit,
    enums.CoalType.Lignite: u.tonne_lignite,
    enums.CoalType.ImportedBituminous: u.tonne_coal_bit,
    enums.CoalType.ImportedSubbituminous: u.tonne_coal_subbit,
}

kilotonne_by_coal_type = {
    enums.CoalType.CanadianBituminous: u.kilotonne_coal_bit,
    enums.CoalType.CanadianSubbituminous: u.kilotonne_coal_subbit,
    enums.CoalType.Lignite: u.kilotonne_lignite,
    enums.CoalType.ImportedBituminous: u.kilotonne_coal_bit,
    enums.CoalType.ImportedSubbituminous: u.kilotonne_coal_subbit,
}

m3_by_roundwood_species_group = {
    enums.RoundwoodSpeciesGroup.Unspecified: u.m3_wood,
    enums.RoundwoodSpeciesGroup.Softwoods: u.m3_softwood,
    enums.RoundwoodSpeciesGroup.Hardwoods: u.m3_hardwood,
}
