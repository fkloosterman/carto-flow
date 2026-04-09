#!/usr/bin/env python3
"""Download US state population estimates from the US Census Bureau.

Coverage
--------
Annual estimates for all 50 states and DC, 1900-2024.
Puerto Rico is included from 2010 onward (absent from pre-2010 sources).
Pre-1970 values are rounded intercensal estimates (nearest thousand).

Data sources
------------
1900-1969  github.com/JoshData/historical-state-population-csv
           (annual intercensal estimates, rounded; no PR)
1970-1979  www2.census.gov … tables/1900-1980/state/asrh/pe-19.csv
           (age/sex/race stratified → aggregated to state totals)
1980-1989  www2.census.gov … tables/1980-1990/state/asrh/st8090ts.txt
1990-1999  www2.census.gov … tables/1990-2000/state/totals/st-99-07.txt
2000-2009  www2.census.gov … datasets/2000-2010/intercensal/state/st-est00int-alldata.csv
2010-2019  www2.census.gov … datasets/2010-2020/state/totals/nst-est2020-alldata.csv
2020-2024  www2.census.gov … datasets/2020-2024/state/totals/NST-EST2024-ALLDATA.csv

Output columns
--------------
year, state_fips, state_name, state_abbr, population

Usage
-----
    python scripts/download_state_population.py
    python scripts/download_state_population.py --output path/to/output.csv
"""

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# State FIPS mappings
# ---------------------------------------------------------------------------

FIPS_TO_ABBR: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "72": "PR",
}

ABBR_TO_FIPS: dict[str, str] = {v: k for k, v in FIPS_TO_ABBR.items()}

FIPS_TO_NAME: dict[str, str] = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "72": "Puerto Rico",
}

_BASE = "https://www2.census.gov/programs-surveys/popest"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    print(f"  GET {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# Era parsers
# ---------------------------------------------------------------------------

_JOSHDATA_URL = (
    "https://raw.githubusercontent.com/JoshData/"
    "historical-state-population-csv/main/historical_state_population_by_year.csv"
)


def _parse_pre1970(text: str) -> pd.DataFrame:
    """historical_state_population_by_year.csv (JoshData/historical-state-population-csv).

    No header; columns: state_abbr, year, population (plain integers).
    Annual intercensal estimates 1900-2025 for 50 states + DC (no PR).
    Pre-1970 values are rounded to the nearest thousand.
    Only years < 1970 are kept; the Census Bureau sources handle 1970-2024.
    """
    df = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=["state_abbr", "year", "population"],
    )
    df = df[df["year"] < 1970].copy()
    df["state_fips"] = df["state_abbr"].map(ABBR_TO_FIPS)
    df = df.dropna(subset=["state_fips"])
    df["population"] = pd.to_numeric(df["population"], errors="coerce").astype(int)
    return df[["year", "state_fips", "population"]]


def _parse_1970s(text: str) -> pd.DataFrame:
    """pe-19.csv: age/sex/race-stratified state estimates, 1970-1979.

    Numbers use comma thousands separators inside a comma-delimited CSV, so
    the commas are stripped before parsing.
    """
    # Strip thousands commas only inside quoted fields (e.g. "105,856" → "105856").
    # Unquoted fields like "169,129,107" are real CSV delimiters and must stay.
    text = re.sub(
        r'"(\d[\d,]+)"',
        lambda m: '"' + m.group(1).replace(",", "") + '"',
        text,
    )

    lines = text.splitlines()

    # Find the header row (contains "Year" and "FIPS" or "State")
    header_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        if "year" in lower and ("fips" in lower or "state" in lower):
            header_idx = i
            break

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')

    year_col = next((c for c in df.columns if "year" in c.lower()), None)
    fips_col = next((c for c in df.columns if "fips" in c.lower()), None)
    if year_col is None or fips_col is None:
        raise ValueError(f"Could not identify Year/FIPS columns in pe-19.csv. Found: {list(df.columns)}")

    # Age-group columns: contain a digit (e.g. "Under 5 years", "5 to 9 years",
    # "85 years and over").  This avoids the false match of "year" inside "years".
    age_cols = [c for c in df.columns if re.search(r"\d", c)]

    for col in age_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    df["population"] = df[age_cols].sum(axis=1)
    df["year"] = pd.to_numeric(df[year_col].str.strip(), errors="coerce").astype("Int64")
    df["state_fips"] = df[fips_col].astype(str).str.strip().str.zfill(2)

    result = df[df["state_fips"].isin(FIPS_TO_ABBR)].groupby(["year", "state_fips"])["population"].sum().reset_index()
    result["population"] = result["population"].astype(int)
    return result


def _parse_1980s(text: str) -> pd.DataFrame:
    """st8090ts.txt: total state population, 1980-1989.

    The file is presented in two separate blocks:
      Block 1: 4/80cen  7/81  7/82  7/83  7/84   (5 columns)
      Block 2: 7/85  7/86  7/87  7/88  7/89  4/90cen  (6 columns)

    Each block starts with a header line containing "/" characters.
    State rows start with a 2-letter abbreviation.
    1990 (April census) is omitted; it will come from the 1990-1999 file.
    """

    def _to_year(h: str) -> int:
        # "4/80cen" → 1980, "7/81" → 1981, "4/90cen" → 1990
        return 1900 + int(h.replace("cen", "").split("/")[1])

    # Split into blocks by finding header lines (contain "/" and digits, no letters)
    blocks: list[tuple[list[int], list[str]]] = []  # [(years, data_lines), ...]
    current_years: list[int] = []
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Header line: every token matches a date pattern like "4/80cen" or "7/81"
        # This avoids false positives from comment lines like "P25-1106, issued 11/93"
        if stripped and all(re.match(r"^\d+/\d+", tok) for tok in stripped.split()):
            if current_years:
                blocks.append((current_years, current_lines))
            current_years = [_to_year(h) for h in stripped.split()]
            current_lines = []
        else:
            current_lines.append(line)

    if current_years:
        blocks.append((current_years, current_lines))

    # Accumulate per-state data across blocks
    state_data: dict[str, dict[int, int]] = {}
    for years, data_lines in blocks:
        for line in data_lines:
            parts = line.split()
            if not parts:
                continue
            abbr = parts[0]
            if abbr not in ABBR_TO_FIPS:
                continue
            fips = ABBR_TO_FIPS[abbr]
            if fips not in state_data:
                state_data[fips] = {}
            for year, val in zip(years, parts[1:], strict=False):
                with contextlib.suppress(ValueError):
                    state_data[fips][year] = int(val.replace(",", ""))

    rows = [
        {"year": year, "state_fips": fips, "population": pop}
        for fips, year_pops in state_data.items()
        for year, pop in year_pops.items()
        if year < 1990  # 1990 covered by next file
    ]
    return pd.DataFrame(rows)


def _parse_1990s(text: str) -> pd.DataFrame:
    """st-99-07.txt: total state population, 1990-1999.

    Fixed-width file.  Columns (left→right): 7/1/99, 7/1/98, ..., 7/1/90,
    4/1/90cen (11 columns).  We take the 10 July-1 estimates (1999 down to
    1990) and match each line to a state by name.

    State names are matched longest-first to resolve ambiguities like
    "Virginia" vs "West Virginia".
    """
    lines = text.splitlines()

    # Years corresponding to the first 10 (July) population columns
    years = list(range(1999, 1989, -1))  # [1999, 1998, …, 1990]

    # Sort state names longest-first to resolve "Virginia" / "West Virginia"
    name_to_fips: dict[str, str] = dict(
        sorted(
            ((name, fips) for fips, name in FIPS_TO_NAME.items()),
            key=lambda x: -len(x[0]),
        )
    )

    rows = []
    for line in lines:
        # All integers on the line (block number, FIPS code, population values)
        raw_nums = re.findall(r"\b\d[\d,]*\b", line)
        if not raw_nums:
            continue

        # The file has multiple "blocks" per state (components of change).
        # Block 1 is the total resident population; skip all others.
        try:
            if int(raw_nums[0].replace(",", "")) != 1:
                continue
        except ValueError:
            continue

        # Extract population figures ≥ 100,000 (excludes block/FIPS code numbers)
        pops = []
        for n in raw_nums:
            try:
                val = int(n.replace(",", ""))
                if val >= 100_000:
                    pops.append(val)
            except ValueError:
                pass

        if len(pops) < 10:
            continue

        # Find the matching state name
        matched_fips = None
        for name, fips in name_to_fips.items():
            if re.search(r"\b" + re.escape(name) + r"\b", line, re.IGNORECASE):
                matched_fips = fips
                break

        if matched_fips is None:
            continue

        for year, pop in zip(years, pops[:10], strict=False):
            rows.append({"year": year, "state_fips": matched_fips, "population": pop})

    return pd.DataFrame(rows)


def _parse_2000s(text: str) -> pd.DataFrame:
    """st-est00int-alldata.csv: intercensal estimates 2000-2009.

    The file is stratified by sex, origin, race, and age group.  Filtering to
    SEX=0, ORIGIN=0, RACE=0, AGEGRP=0 gives the all-group total population.
    2010 is omitted here; it will come from the 2010-2020 file.
    """
    df = pd.read_csv(io.StringIO(text))
    mask = (df["SEX"] == 0) & (df["ORIGIN"] == 0) & (df["RACE"] == 0) & (df["AGEGRP"] == 0) & (df["STATE"] > 0)
    df = df[mask].copy()
    df["state_fips"] = df["STATE"].astype(str).str.zfill(2)
    df = df[df["state_fips"].isin(FIPS_TO_ABBR)]

    rows = []
    for col in df.columns:
        if not col.startswith("POPESTIMATE"):
            continue
        year = int(col.replace("POPESTIMATE", ""))
        if year >= 2010:
            continue
        rows.append(df[["state_fips", col]].rename(columns={col: "population"}).assign(year=year))

    return pd.concat(rows, ignore_index=True)[["year", "state_fips", "population"]]


def _parse_2010s(text: str) -> pd.DataFrame:
    """nst-est2020-alldata.csv: annual estimates 2010-2019.

    SUMLEV=40 selects state-level rows.
    2020 is omitted here; it will come from the 2020-2024 file.
    """
    df = pd.read_csv(io.StringIO(text))
    df = df[df["SUMLEV"] == 40].copy()
    df["state_fips"] = df["STATE"].astype(str).str.zfill(2)
    df = df[df["state_fips"].isin(FIPS_TO_ABBR)]

    rows = []
    for col in df.columns:
        if not col.startswith("POPESTIMATE"):
            continue
        year = int(col.replace("POPESTIMATE", ""))
        if year >= 2020:
            continue
        rows.append(df[["state_fips", col]].rename(columns={col: "population"}).assign(year=year))

    return pd.concat(rows, ignore_index=True)[["year", "state_fips", "population"]]


def _parse_2020s(text: str) -> pd.DataFrame:
    """NST-EST2024-ALLDATA.csv: annual estimates 2020-2024.

    SUMLEV=40 selects state-level rows.
    """
    df = pd.read_csv(io.StringIO(text))
    df = df[df["SUMLEV"] == 40].copy()
    df["state_fips"] = df["STATE"].astype(str).str.zfill(2)
    df = df[df["state_fips"].isin(FIPS_TO_ABBR)]

    rows = []
    for col in df.columns:
        if not col.startswith("POPESTIMATE"):
            continue
        year = int(col.replace("POPESTIMATE", ""))
        rows.append(df[["state_fips", col]].rename(columns={col: "population"}).assign(year=year))

    return pd.concat(rows, ignore_index=True)[["year", "state_fips", "population"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output: Path) -> None:
    eras = [
        ("1900-1969", _JOSHDATA_URL, _parse_pre1970),
        ("1970-1979", f"{_BASE}/tables/1900-1980/state/asrh/pe-19.csv", _parse_1970s),
        ("1980-1989", f"{_BASE}/tables/1980-1990/state/asrh/st8090ts.txt", _parse_1980s),
        ("1990-1999", f"{_BASE}/tables/1990-2000/state/totals/st-99-07.txt", _parse_1990s),
        ("2000-2009", f"{_BASE}/datasets/2000-2010/intercensal/state/st-est00int-alldata.csv", _parse_2000s),
        ("2010-2019", f"{_BASE}/datasets/2010-2020/state/totals/nst-est2020-alldata.csv", _parse_2010s),
        ("2020-2024", f"{_BASE}/datasets/2020-2024/state/totals/NST-EST2024-ALLDATA.csv", _parse_2020s),
    ]

    pieces = []
    for label, url, parser in eras:
        print(f"Downloading {label}…")
        try:
            pieces.append(parser(_fetch(url)))
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    df = pd.concat(pieces, ignore_index=True)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["state_name"] = df["state_fips"].map(FIPS_TO_NAME)
    df["state_abbr"] = df["state_fips"].map(FIPS_TO_ABBR)
    df = (
        df.sort_values(["state_fips", "year"])
        .drop_duplicates(subset=["year", "state_fips"], keep="first")
        .reset_index(drop=True)
    )
    df["population"] = df["population"].astype(int)
    df = df[["year", "state_fips", "state_name", "state_abbr", "population"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nWrote {len(df):,} rows → {output}")
    print(f"Years:              {df['year'].min()}-{df['year'].max()}")
    print(f"States/territories: {df['state_abbr'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/us_state_population_1900_2024.csv"),
        help="Output CSV path (default: data/us_state_population_1900_2024.csv)",
    )
    main(parser.parse_args().output)
