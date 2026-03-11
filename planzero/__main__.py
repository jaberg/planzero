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
    parser_cache_petrinex.add_argument('--include_ghgrp', default=False, help='include contributions from ghgrp-listed facilities')
    parser_cache_petrinex.set_defaults(func=cache_petrinex)

    args = parser.parse_args()
    args.func(args)
