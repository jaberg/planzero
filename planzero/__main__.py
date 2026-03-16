import argparse

def print_max_gaps(args):
    from . import est_nir
    assert args.year == 2005
    est = est_nir.EstSectorEmissions()
    max_gap = est.max_gap_2005()


def cache_pollution_waste_canada_report_details(args):
    import requests
    from . import pollution_waste_canada
    from . import ghgrp
    df = ghgrp._read_emissions_sources() # 2022 & 2023
    npri_ids = df[ghgrp.ESKey.NPRI_ID]
    for npri_id in sorted(npri_ids.unique()):
        if npri_id > 0:
            try:
                pollution_waste_canada.report_details(
                    int(npri_id),
                    args.report_year,
                    read_cache=True,
                    write_cache=True)
            except requests.exceptions.HTTPError as err:
                print(err)


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
    #parser.add_argument('--foo', action='store_true', help='foo help')
    subparsers = parser.add_subparsers(help='subcommand help')

    parser_print_max_gap = subparsers.add_parser('print_max_gaps')
    parser_print_max_gap.add_argument('--year', type=int, default=2005, help='year')
    parser_print_max_gap.set_defaults(func=print_max_gaps)

    parser_cache_pollution_details = subparsers.add_parser('cache_pollution_waste_canada_report_details')
    parser_cache_pollution_details.add_argument('--report-year', default=2022, help='report year')
    parser_cache_pollution_details.set_defaults(func=cache_pollution_waste_canada_report_details)

    parser_cache_petrinex = subparsers.add_parser('cache_petrinex')
    parser_cache_petrinex.add_argument('--year', help='report year')
    parser_cache_petrinex.add_argument('--PT', help='report year')
    parser_cache_petrinex.add_argument('--include_ghgrp', action='store_true', help='include contributions from ghgrp-listed facilities')
    parser_cache_petrinex.set_defaults(func=cache_petrinex)

    parser_request_all_pages = subparsers.add_parser('request_all_pages')
    parser_request_all_pages.set_defaults(func=request_all_planzero_pages)

    args = parser.parse_args()
    args.func(args)
