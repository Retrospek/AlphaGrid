import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


def get_mappings():
    # Sector Conversion Map
    sector_mapping = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLE': 'Energy',
            'XLV': 'Health Care',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }

    # GNN Mapping
    node_mapping = {}
    inverse_node_mapping = {}

    for i, key in enumerate(sector_mapping.keys()):
        node_mapping[i] = key
        inverse_node_mapping[key] = i

    return sector_mapping, node_mapping, inverse_node_mapping

def download_sector_data(sector_mapping) -> pd.DataFrame:
        start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading Sector-Based Data...")
        sector_data = yf.download(tickers=list(sector_mapping.keys()), start=start_date, end=end_date)
        print(f"Downloaded yfinance sector data for =:\n-> Keys: {list(sector_mapping.keys())}\n-> Values: list{sector_mapping.values()}")

        print(f"Flattening Column Labels to Remove Tuple-Based Column Structure... & Shifting Date into Column")
        sector_data.columns = [f"{sector}_{metric}" for metric, sector in sector_data.columns]
        sector_data = sector_data.reset_index()

        print(f"* Done Preprocessing and Organizing Data *")
        return sector_data

# Create a list of Different Dataframes
def get_sector_data_separated(sector_mapping, original_data, original_columns) -> dict:
    sector_separated_data = {}

    for sector in sector_mapping.keys():
        sector_separated_data[sector] = original_data[[f"{sector}_{metric[metric.index("_") + 1:]}" for metric in original_columns if sector in metric]].copy()
        sector_data = sector_separated_data[sector] # Dynamic State any change conserved
        new_columns = [old[old.index("_") + 1:] for old in sector_data.columns if "_" in old]
        sector_data.columns = new_columns
        sector_data["Date"] = original_data["Date"].values
        sector_data["Sector"] = sector
        sector_data["Liquidity"] = sector_data["Volume"] * sector_data["Close"]

        sector_data["IntraDay_Return($)"] = sector_data["Close"] - sector_data["Open"]
        sector_data["InterDay_Return($)"] = sector_data["Close"] - sector_data["Close"].shift(1)

        sector_data["IntraDay_Return(%)"] = sector_data["IntraDay_Return($)"] / sector_data["Open"]
        sector_data["InterDay_Return(%)"] = sector_data["InterDay_Return($)"] / sector_data["Close"].shift(1)

        for bound in ["Intra", "Inter"]:
            sector_data[f"{bound}_Gain"] = np.max(sector_data["IntraDay_Return($)"], 0)
            sector_data[f"{bound}_Loss"] = np.max(-1 * sector_data["IntraDay_Return($)"], 0)
    return sector_separated_data

def remove_nans(sector_separated_data):
    for sector in sector_separated_data.keys():
        sector_data = sector_separated_data[sector]
        sector_separated_data[sector] = sector_data.dropna()
        print(f"{sector} data length: {len(sector_separated_data[sector])}")
    return sector_separated_data

def align_on_common_dates(sector_separated_data):
    # Find intersection of valid dates across all sectors
    all_dates = [set(df["Date"]) for df in sector_separated_data.values()]
    common_dates = set.intersection(*all_dates)

    for sector in sector_separated_data:
        df = sector_separated_data[sector]
        df = df[df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True)
        sector_separated_data[sector] = df

    return sector_separated_data

def define_target(sector_separated_data):
    epsilon = 1e-6
    for sector in sector_separated_data.keys():
        sector_data = sector_separated_data[sector]
        delta = sector_data["RV_5Days_1DayDelta(%)"].shift(-1)
        sector_data["Target"] = delta #np.sign(delta) * np.log(np.abs(delta) + epsilon)

    return sector_separated_data

def check_download_uniformity(sector_data, sector_mapping):
    """Check if all sectors actually have the same dates after download"""
    print("\n=== Checking Download Uniformity ===")
    
    # Extract dates for each sector from the flattened data
    sector_dates = {}
    for sector in sector_mapping.keys():
        # Get all columns for this sector
        sector_cols = [col for col in sector_data.columns if col.startswith(f"{sector}_")]
        if sector_cols:
            sector_subset = sector_data[["Date"] + sector_cols].dropna()
            sector_dates[sector] = set(sector_subset["Date"])
            print(f"{sector}: {len(sector_subset)} rows with data")
    
    # Check if all have same dates
    all_date_sets = list(sector_dates.values())
    if len(set(len(dates) for dates in all_date_sets)) == 1:
        print("✓ All sectors have same number of dates")
    else:
        print("✗ Sectors have different numbers of dates!")
        for sector, dates in sector_dates.items():
            print(f"  {sector}: {len(dates)} dates")
    
    # Check intersection
    common_dates = set.intersection(*all_date_sets)
    print(f"Common dates: {len(common_dates)}")
    
    return sector_dates