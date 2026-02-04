from stats_can.sc import (pathlib, pd, zipfile, parse_tables, download_tables)
import stats_can
import markdown
import pandas as pd

def sc_metadata_by_product_id(productId):
    return stats_can.schemas.CubeMetadata(**stats_can.scwds.get_cube_metadata(productId)[0])


def markdown_from_dimension(scdim):
    lines = []
    # print(scdim)
    lines.append(f'## Dimension {scdim["dimensionPositionId"]}: {scdim["dimensionNameEn"]}')
    member_by_id = {member['memberId']: member for member in scdim['member']}
    children_by_id = {member['memberId']: [] for member in scdim['member']}
    printed_by_id = {member['memberId']: False for member in scdim['member']}
    indent_by_id = {member['memberId']: 0 for member in scdim['member']}

    for mid, member in member_by_id.items():
        if member['parentMemberId'] is not None:
            children_by_id[member['parentMemberId']].append(mid)
            indent_by_id[mid] = indent_by_id[member['parentMemberId']] + 2
    def append_children(_mid):
        for cid in children_by_id[_mid]:
            if not printed_by_id[cid]:
                lines.append(' ' * indent_by_id[cid] + '* ' + member_by_id[cid]['memberNameEn'])
                printed_by_id[cid] = True
                append_children(cid)

    for mid, member in member_by_id.items():
        if not printed_by_id[mid]:
            lines.append(' ' * indent_by_id[mid] + '* ' + member_by_id[mid]['memberNameEn'])
            printed_by_id[mid] = True
            append_children(mid)
    return '\n'.join(lines)


def jupyter_display_metadata(metadata):
    # import inside function so module can be imported without IPython installed
    from IPython.display import display, HTML
    md_text = (
        f'#{metadata["cubeTitleEn"]}'
        f' ([Stats-Can: {metadata["productId"]}](https://https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid={metadata["productId"]}))\n'
        + '\n\n'.join(markdown_from_dimension(dim) for dim in metadata['dimension']))
    return display(HTML(markdown.markdown(md_text)))


def zip_table_to_dataframe(
    table: str, path: pathlib.Path | None = None
) -> pd.DataFrame:
    """Read a StatsCan table into a pandas DataFrame.

    If a zip file of the table does not exist in path, downloads it

    Parameters
    ----------
    table
        the table to load to dataframe from zipped csv
    path
        where to download the tables or load them, default will go to current working directory

    Returns
    -------
    :
        the table as a dataframe
    """
    path = pathlib.Path(path) if path else pathlib.Path()
    # Parse tables returns a list, can only do one table at a time here though
    table = parse_tables(table)[0]
    table_zip = table + "-eng.zip"
    table_zip = path / table_zip
    if not table_zip.is_file():
        download_tables([table], path)
    csv_file = table + ".csv"
    with zipfile.ZipFile(table_zip) as myzip:
        with myzip.open(csv_file) as myfile:
            col_names = pd.read_csv(myfile, nrows=0).columns
        # reopen the file or it misses the first row
        with myzip.open(csv_file) as myfile:
            types_dict = {"VALUE": float}
            types_dict.update({col: str for col in col_names if col not in types_dict})
            df = pd.read_csv(myfile, dtype=types_dict)

    possible_cats = [
        "GEO",
        "DGUID",
        "STATUS",
        "SYMBOL",
        "TERMINATED",
        "DECIMALS",
        "UOM",
        "UOM_ID",
        "SCALAR_FACTOR",
        "SCALAR_ID",
        "VECTOR",
        "COORDINATE",
        "Wages",
        "National Occupational Classification for Statistics (NOC-S)",
        "Supplementary unemployment rates",
        "Sex",
        "Age group",
        "Labour force characteristics",
        "Statistics",
        "Data type",
        "Job permanency",
        "Union coverage",
        "Educational attainment",
    ]
    actual_cats = [col for col in possible_cats if col in col_names]
    df[actual_cats] = df[actual_cats].astype("category")
    # df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], errors="ignore") # JB: leads to assertion error so commenting it out
    return df


