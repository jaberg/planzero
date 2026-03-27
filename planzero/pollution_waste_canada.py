import datetime
import functools
import json
import os

import pydantic
import requests
from pathlib import Path

cache_dir = 'cache/pollution_waste_canada'

class ProgramElem(pydantic.BaseModel):
    identifier: str
    url: str | None
    en: str
    fr: str


class Facility(pydantic.BaseModel):
    npriId: int
    ghgrpId: str | None
    name: str
    portable: bool
    isCase3: bool
    isCase4: bool
    numberOfEmployees: int
    industry: dict
    website: str | None
    physicalAddress: dict
    geographicAddress: dict
    operatingSchedule: dict | None
    activities: list
    otherPrograms: list[ProgramElem]
    permits: list
    formattedAddress: dict

    def petrinex_ids(self):
        return [elem.identifier for elem in self.otherPrograms
                if 'Petrinex' in elem.en]

    def ghgrp_ids(self):
        return [elem.identifier for elem in self.otherPrograms
                if 'GHGRP' in elem.en]

    def ghgrp_id(self):
        rval, = self.ghgrp_ids()
        return rval


class Details(pydantic.BaseModel):
    comments: list
    company: dict
    contacts: list
    disposalsTransfers: list
    egus: list
    facility: Facility
    ghgrpId: str | None = None # can also be found in facility['ghgrpId']
    npriId: int
    otherYears: list
    p2Plan: dict
    releases: list
    reportTypes: list
    reportYear: int
    scpDetails: list
    submissionDate: datetime.datetime
    substanceSummary: list


def fetch_report_details(npri_id, report_year):
    """
    Fetches facility details directly from the NPRI API
    """

    api_url = f"https://pollution-waste.canada.ca/sradapi/v2/npri/Details/Report?npriId={npri_id}&reportYear={report_year}"
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    response = requests.get(api_url, headers=headers, timeout=15)
    response.raise_for_status()
    return Details( **response.json())


class NoDetailsAvailable(Exception):
    pass


def report_details(npri_id, report_year,
                   read_cache=True,
                   fetch=True,
                   write_cache=True):
    """
    Attempts to load Details from a local JSON cache. If missing, downloads
    via fetch_report_details and updates the cache.
    """
    # Create a unique filename based on ID and Year
    cache_path = Path(cache_dir) / f"details_{npri_id}_{report_year}.json"

    # 1. Attempt to load from cache
    if read_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                #print(f"Loading NPRI {npri_id} ({report_year}) from cache...")
                return Details.model_validate(cached_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Cache corrupted for {npri_id}, re-downloading... Error: {e}")

    # 2. Cache missing or invalid: Fetch new data
    if fetch:
        print(f"Cache miss. Fetching details for NPRI {npri_id} from API...")
        details_obj = fetch_report_details(npri_id, report_year)
    else:
        raise NoDetailsAvailable(npri_id, report_year)

    # 3. Save to cache
    if write_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            # model_dump_json() ensures Pydantic types are serialized correctly
            f.write(details_obj.model_dump_json(indent=4))

    return details_obj


class GHGRP_by_Petrinex_Mapping(pydantic.BaseModel):
    year: int
    ghgrp_id_by_petrinex_id: dict[str, str]


@functools.cache
def ghgrp_id_by_petrinex_id(report_year=2022):
    # see __main__.py cache_ghgrp_by_petrinex for function that writes this file
    cache_path = Path(cache_dir) / f"ghgrp_id_by_petrinex_id_{report_year}.json"
    with open(cache_path, "r", encoding="utf-8") as f:
        cached_data = json.load(f)
        #print(f"Loading NPRI {npri_id} ({report_year}) from cache...")
        return GHGRP_by_Petrinex_Mapping.model_validate(cached_data)
