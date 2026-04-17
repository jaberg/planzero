import os
import sys

# Ensure the planzero package can be imported
sys.path.append(os.getcwd())

# Enable disk caching
os.environ['PLANZERO_USE_DISK_CACHE'] = '1'
os.environ['PLANZERO_CACHE_DIR'] = '.planzero_cache'

import planzero
import planzero.scenarios
import planzero.sim
import planzero.est_nir
from planzero.enums import IPCC_Sector

def warmup():
    print("Starting warmup to populate disk cache...")
    
    # Iterate through all available scenarios
    scenario_names = list(planzero.scenarios.scenarios.keys())
    print(f"Scenarios to process: {scenario_names}")
    
    for name in scenario_names:
        print(f"--- Processing scenario: {name} ---")
        try:
            # Trigger full simulation and ablations
            sim_result = planzero.sim.sim_scenario(name)
            
            # Trigger sectoral chart generation (computed_field)
            print(f"Generating main IPCC chart for {name}...")
            _ = sim_result.by_ipcc_sector
            
            # Trigger specific IPCC sector charts
            # We'll do a few major ones to ensure common views are fast
            # Agriculture, Energy, etc.
            sectors_to_warm = [
                'Agriculture/Enteric Fermentation',
                'Stationary Combustion Sources/Public Electricity and Heat Production',
                'Transport/Road Transportation/Heavy-Duty Diesel Vehicles',
                'Fugitive Sources/Oil and Natural Gas/Venting',
            ]
            for catpath in sectors_to_warm:
                print(f"Generating sector chart for {catpath} in {name}...")
                try:
                    _ = sim_result.echart_ipcc_sector(catpath)
                except Exception as e:
                    print(f"Warning: Could not generate chart for {catpath}: {e}")
                    
        except Exception as e:
            print(f"Error processing scenario {name}: {e}")
            import traceback
            traceback.print_exc()

    print("Warmup calling est_nir.EstSectorEmissions...")
    planzero.est_nir.EstSectorEmissions().max_gap_2005()

    print("Warmup complete. Disk cache populated.")

if __name__ == "__main__":
    warmup()
