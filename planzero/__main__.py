import argparse
import os

from pathlib import Path

def print_max_gaps(args):
    from . import est_nir
    assert args.year == 2005
    est = est_nir.EstSectorEmissions()
    max_gap = est.max_gap_2005()


def cache_ghgrp_by_petrinex(args):
    import requests
    from . import pollution_waste_canada
    from . import ghgrp
    df = ghgrp._read_emissions_sources() # 2022 & 2023
    npri_ids = df[ghgrp.ESKey.NPRI_ID]

    cache_path = Path(pollution_waste_canada.cache_dir) / f"ghgrp_id_by_petrinex_id_{args.report_year}.json"
    ghgrp_id_by_petrinex_id = {}
    n_skips = 0
    for npri_id in sorted(npri_ids.unique()):
        if npri_id > 0:
            try:
                details = pollution_waste_canada.report_details(
                    int(npri_id),
                    args.report_year,
                    read_cache=True,
                    fetch=True,
                    write_cache=True)
            except pollution_waste_canada.NoDetailsAvailable as err:
                # TODO: log level
                #print('No details available', npri_id, err)
                n_skips += 1
                continue
            except requests.HTTPError as err:
                # TODO: log level
                #print('HTTP Error', npri_id, err)
                n_skips += 1
                continue
            gids = details.facility.ghgrp_ids()
            if gids:
                gid, = gids
                for pid in details.facility.petrinex_ids():
                    ghgrp_id_by_petrinex_id[pid] = gid
    if n_skips:
        print(f'ghgrp_id_by_petrinex_id skipped {n_skips} NPRI facilities mentioned in GHGRP')

    os.makedirs(pollution_waste_canada.cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        obj = pollution_waste_canada.GHGRP_by_Petrinex_Mapping(
            year=args.report_year,
            ghgrp_id_by_petrinex_id=ghgrp_id_by_petrinex_id)
        f.write(obj.model_dump_json(indent=1))



def cache_petrinex(args):
    from . import petrinex
    petrinex.main_build_cache(args)


def request_all_planzero_pages(args):
    import requests
    import time

    class SiteClient(object):
        def get(self, path):
            t0 = time.time()
            rval = requests.get(f'https://planzero.ca{path}')#, timeout=30)
            t1 = time.time()
            self.last_get_time = t1 - t0
            return rval.status_code

    client = SiteClient()
    for path in [
        '/',
        '/ipcc-sectors/',
        '/strategies/',
        '/about/',
    ]:
        status_code = client.get(path)
        print(status_code, '{:.2f}'.format(client.last_get_time), path)

    from . import ipcc_canada
    for catpath in ipcc_canada.catpaths:
        path = f'/ipcc-sectors/{catpath}/'
        status_code = client.get(path)
        print(status_code, '{:.2f}'.format(client.last_get_time), path)

    from . import blog
    for url_filename in blog._blogs_by_url_filename:
        path = f'/blog/{url_filename}/'
        status_code = client.get(path)
        print(status_code, '{:.2f}'.format(client.last_get_time), path)


if __name__ == '__main__':

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='planzero')
    subparsers = parser.add_subparsers(help='subcommand help')

    parser_print_max_gap = subparsers.add_parser('print_max_gaps')
    parser_print_max_gap.add_argument('--year', type=int, default=2005, help='year')
    parser_print_max_gap.set_defaults(func=print_max_gaps)

    parser_cache_ghgrp_by_petrinex = subparsers.add_parser('cache_ghgrp_by_petrinex')
    parser_cache_ghgrp_by_petrinex.add_argument('--report-year', default=2022, help='report year')
    parser_cache_ghgrp_by_petrinex.set_defaults(func=cache_ghgrp_by_petrinex)

    parser_cache_petrinex = subparsers.add_parser('cache_petrinex')
    parser_cache_petrinex.add_argument('--year', help='report year')
    parser_cache_petrinex.add_argument('--PT', help='report year')
    parser_cache_petrinex.add_argument('--include_ghgrp', action='store_true', help='include contributions from ghgrp-listed facilities')
    parser_cache_petrinex.add_argument('--large-emitter-cutoff-monthly',
                                       type=float,
                                       default=None, help='kt_CO2e')
    parser_cache_petrinex.set_defaults(func=cache_petrinex)

    parser_request_all_pages = subparsers.add_parser('request_all_pages')
    parser_request_all_pages.set_defaults(func=request_all_planzero_pages)

    args = parser.parse_args()
    args.func(args)
