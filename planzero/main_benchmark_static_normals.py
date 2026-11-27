
import datetime
import platform
import subprocess
import time

import psutil

from .enums import GHG, IPCC_Sector
from .nir_model import NIR_ScenarioModel
from .nir_static_normals_model import (
    StaticNormals_ModelElement,
)
from .scenario_model import MCMC_Arguments, compute_model


def get_cpu_info():
    try:
        # On Linux/Colab, we can parse /proc/cpuinfo
        cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | uniq", shell=True).decode().strip()
        cpu_info = cpu_info.split(":")[1].strip()
    except Exception:
        cpu_info = platform.processor() or "Unknown CPU"
    return cpu_info

def main(args):
    print("=== System Characterization for Benchmarking ===")
    print(f"OS / Kernel:   {platform.system()} {platform.release()}")
    print(f"CPU Model:     {get_cpu_info()}")
    print(f"Physical Cores:{psutil.cpu_count(logical=False)}")
    print(f"Total Cores:   {psutil.cpu_count(logical=True)}")
    print(f"Total Memory:  {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available Mem: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("=================================================")


    model = NIR_ScenarioModel(
            data_cutoff=datetime.date(2024, 12, 31),
            last_forecast_year=2030,
            num_warmup=2,
            num_samples=2,
            expect_all_sectors=False)

    snorm_elem = StaticNormals_ModelElement(
                sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
                ghg=GHG.CO2)
    model.add_element(snorm_elem)
            #mcmc_group=f'{sector.value}_{ghg.value}')

    model.add_emissions_sums()
    model.add_element_prenir_2025_m04()

    model.MCMC_args = MCMC_Arguments(
            num_warmup=model.MCMC_args.num_warmup,
            num_samples=model.MCMC_args.num_samples,
            thinning=model.MCMC_args.thinning,
            n_divergences_acceptable=int(1e10),
            NUTS_args=model.MCMC_args.NUTS_args)

    # this test used to compute the model, but it has been cut
    # to attempt to save memory
    t0 = time.time()
    compute_model(model, seed_or_key=1)
    t1 = time.time()
    print(f'compute_model_{model.MCMC_args.num_warmup}_{model.MCMC_args.num_samples} seconds: {(t1 - t0):.3f}')

    model.MCMC_args = MCMC_Arguments(
            num_warmup=model.MCMC_args.num_warmup,
            num_samples=200,
            thinning=model.MCMC_args.thinning,
            n_divergences_acceptable=int(1e10),
            NUTS_args=model.MCMC_args.NUTS_args)
    compute_model(model, seed_or_key=1)
    t2 = time.time()
    print(f'compute_model_{model.MCMC_args.num_warmup}_{model.MCMC_args.num_samples} seconds: {(t2 - t1):.3f}')

    print("\n")
    print("both times should be under about 15 seconds")
    print("the second time should not be much bigger than the first, because compilation should still dominate the total")
