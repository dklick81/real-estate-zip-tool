#!/usr/bin/env python3
"""
build_real_zip_scores.py

Standalone ETL script to produce `real_zip_scores.csv` by:
  - Fetching ZIP metadata
  - Merging Zillow ZHVI data
  - Pulling Census ACS metrics
  - Aggregating Cleveland crime incidents
  - Retrieving SchoolDigger scores
  - Imputing missing values
  - Computing normalized core metrics
"""

#importing required libraries
import os
from tqdm import tqdm
import toml
import pandas as pd
import requests
import json
import warnings
from requests.exceptions import JSONDecodeError
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

#configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

#functions for Census data and school quality data
def get_census_data(zcta_list, api_key):
    vars = [
        "B19013_001E",  # median household income
        "B08303_001E",  # mean commute time
        "B25088_001E",  # median real estate taxes paid
        "B01003_001E"   # total population
    ]
    all_chunks = []
    # split into batches of 50 ZCTAs per request
    for i in tqdm(range(0, len(zcta_list), 50), desc="Census batches", unit="batch"):
        chunk = zcta_list[i:i+50]
        url = (
            f"https://api.census.gov/data/2021/acs/acs5?get=NAME,{','.join(vars)}"
            f"&for=zip%20code%20tabulation%20area:{','.join(chunk)}&key={api_key}"
        )
        try:
            resp = requests.get(url)
        except Exception as e:
            logger.error(f"Census request exception for batch {i//50+1}: {e}")
            continue        
        if resp.status_code != 200:
            logger.error(f"Census batch {i//50+1} failed: HTTP {resp.status_code}")
            continue
        try:
            data = resp.json()
        except (ValueError, JSONDecodeError):
            logger.warning(f"Failed to parse Census JSON for batch {i//50+1}: {e}")
            continue
        cols = data[0]
        rows = data[1:]
        df_chunk = pd.DataFrame(rows, columns=cols)
        for v in vars:
            df_chunk[v] = pd.to_numeric(df_chunk[v], errors="coerce")
        df_chunk = df_chunk.rename(columns={
            "zip code tabulation area": "zip_code",
            "B19013_001E": "median_income",
            "B08303_001E": "mean_commute",
            "B25088_001E": "median_property_tax",
            "B01003_001E": "population"
        })
        all_chunks.append(df_chunk)
    if all_chunks:
        return pd.concat(all_chunks, ignore_index=True)
    else:
        # return empty DataFrame with expected columns
        cols = ['zip_code', 'median_income', 'mean_commute', 'median_property_tax', 'population']
        return pd.DataFrame(columns=cols)


def get_school_quality(zip_code, app_id, app_key):
    """
    Returns the mean 'averageStandardScore' for all schools in `zip_code`.
    Falls back to None if no data.
    """
    url = "https://api.schooldigger.com/v2.0/schools"
    params = {
        "st": "OH",            # or derive from zip_code if needed
        "zip": zip_code,
        "appID": app_id,
        "appKey": app_key,
        "version": "2.0"
    }
    try:
        resp = requests.get(url, params=params)
    except Exception as e:
        logger.error(f"SchoolDigger request exception for ZIP {zip_code}: {e}")
        return None
    if resp.status_code != 200:
        logger.error(f"SchoolDigger lookup failed for ZIP {zip_code}: HTTP {resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as e:
        logger.warning(f"Failed to parse SchoolDigger JSON for ZIP {zip_code}: {e}")
        return None
    scores = []
    for school in data.get("schoolList", []):
        history = school.get("rankHistory", [])
        if history:
            # grab the latest averageStandardScore
            score = history[0].get("averageStandardScore")
            if score is not None:
                scores.append(score)

    # return the ZIP‐wide average
    return sum(scores) / len(scores) if scores else None

def main():
    #configuring data file parameters and setting up API keys
    CENSUS_API_KEY = "88049c4b2a34868b8f45036c460d71d96894e541"
    SCHOOLDIGGER_APP_ID = "ec7cff4b"
    SCHOOLDIGGER_API_KEY = "2277f7a4a95ce0808ad65dd8a87340a6"
    DATA_DIR = r"C:\Users\Dan\Documents\capstone\raw data"
    ZIP_FILE   = os.path.join(DATA_DIR, "uszips.csv")
    ZHVI_FILE  = os.path.join(DATA_DIR, "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
    CRIME_FILE = os.path.join(DATA_DIR, 'cleveland_crime_incidents.csv')
    OUTPUT_CSV = os.path.join(DATA_DIR, "real_zip_scores.csv")

    #secure API key loading
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"Secrets file not found at {secrets_path}")
    creds = toml.load(secrets_path)
    CENSUS_API_KEY         = creds['Census']['API_KEY']
    SCHOOLDIGGER_APP_ID    = creds['SchoolDigger']['APP_ID']
    SCHOOLDIGGER_API_KEY   = creds['SchoolDigger']['APP_KEY']

    #loading ZIP coordinates
    try:
        zips = pd.read_csv(ZIP_FILE, dtype={"zip": str})
    except Exception as e:
        logger.error(f"Failed to load ZIP metadata: {e}")
        return
    zips = zips[zips["state_id"] == "OH"][ ["zip", "lat", "lng", "population", "density", "county_fips"] ]
    zips = zips.rename(columns={"zip":"zip_code","lat":"latitude","lng":"longitude","density":"pop_density","population":"zip_population"})

    #loading Zillow ZHVI data
    try:
        zhvi = pd.read_csv(ZHVI_FILE, dtype={"RegionName": str})
    except Exception as e:
        logger.error(f"Failed to load ZHVI data: {e}")
        return
    zhvi = zhvi.rename(columns={"RegionName":"zip_code"})
    recent_col = zhvi.columns[-1]
    zhvi = zhvi[["zip_code", recent_col]].rename(columns={recent_col: "zhvi"})

    #fetching Census data
    zcta_list = zips["zip_code"].tolist()
    census_df = get_census_data(zcta_list, CENSUS_API_KEY)

    #merge
    df = zips.merge(census_df, on="zip_code", how="left").merge(zhvi, on="zip_code", how="left")
    df = df.dropna(subset=["median_income", "zhvi", "mean_commute"]).reset_index(drop=True)

    #add crime rate
    if os.path.exists(CRIME_FILE):
        try:
            crime_df = pd.read_csv(CRIME_FILE, dtype={"Zip": str})
        except Exception as e:
            logger.error(f"Failed to load crime data: {e}")
            return
        crime_counts = crime_df.groupby("Zip").size().reset_index(name="crime_count")
        crime_counts.rename(columns={"Zip":"zip_code"}, inplace=True)
        df = df.merge(crime_counts, on="zip_code", how="left")
        df["crime_rate"] = df["crime_count"]/df["zip_population"]

    # impute missing crime_rate with county-level median, then overall median
        county_crime_medians = df.groupby('county_fips')['crime_rate'].median()
        df['crime_rate'] = df.apply(
                lambda r: county_crime_medians[r['county_fips']] if pd.isna(r['crime_rate']) else r['crime_rate'],
                axis=1
            )
        df['crime_rate'].fillna(df['crime_rate'].median(), inplace=True)
    else:
            # No ZIP column detected
            df['crime_rate'] = 0.0

    #add school quality
    print("Fetching school quality for each ZIP (this may take a few minutes)...")
    df["school_quality"] = [
        get_school_quality(z, SCHOOLDIGGER_APP_ID, SCHOOLDIGGER_API_KEY)
        for z in tqdm(df["zip_code"], desc="SchoolDigger lookups", unit="ZIP")
    ]

    # Impute missing school_quality with county-level median, then overall median
    county_medians = df.groupby('county_fips')['school_quality'].median()
    df['school_quality'] = df['school_quality'].fillna(df['county_fips'].map(county_medians))
    overall_median = df['school_quality'].median()
    df['school_quality'].fillna(overall_median, inplace=True)

    #metric computation
    df["affordability"]   = df["median_income"] / df["zhvi"]
    df["property_tax"]    = df["median_property_tax"]
    df["walkability"]     = df["pop_density"]
    df["commute_time"]    = df["mean_commute"]

    #saving dataset
    metrics = ["affordability","property_tax","school_quality","crime_rate","walkability","commute_time"]
    # Ensure all metrics exist (set placeholder if needed)
    if "crime_rate" not in df.columns:
        df["crime_rate"] = 0.0
    if "commute_time" not in df.columns:
        df["commute_time"] = 0.0
    if "affordability" not in df.columns:
        df["affordability"] = 0.0
    if "property_tax" not in df.columns:
        df["property_tax"] = 0.0
    if "school_quality" not in df.columns:
        df["school_quality"] = 0.0
    if "walkability" not in df.columns:
        df["walkability"] = 0.0

    # Output to CSV
    df_output = df[["zip_code","latitude","longitude"] + metrics]
    try:
        df_output.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved real ZIP scores to {OUTPUT_CSV}")
    except Exception as e:
        logger.error(f"Failed to write output CSV {OUTPUT_CSV}: {e}")

if __name__ == '__main__':
    main()