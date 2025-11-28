import pandas as pd
import numpy as np
import os
import yaml

def load_config(config_path="config.yaml"):
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def clean_data():
    """
    Main function to clean and preprocess the Algeria dataset.
    1. Loads raw data.
    2. Filters for key socio-economic indicators.
    3. Transposes data to wide format (Year as index).
    4. Cleans numeric columns and handles missing values.
    5. Saves processed data.
    """
    config = load_config()
    
    raw_path = config["paths"]["raw_data"]
    processed_path = config["paths"]["processed_data"]
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Define Key Indicators (The "Gold" List)
    # We map the complex Indicator Names to simple, usable column names.
    indicators_map = {
        "GDP (current US$)": "GDP_USD",
        "GDP growth (annual %)": "GDP_Growth",
        "Population, total": "Population",
        "Population growth (annual %)": "Population_Growth",
        "CO2 emissions (kt)": "CO2_Emissions",
        "Inflation, consumer prices (annual %)": "Inflation",
        "Life expectancy at birth, total (years)": "Life_Expectancy",
        "Unemployment, total (% of total labor force) (modeled ILO estimate)": "Unemployment",
        "Exports of goods and services (% of GDP)": "Exports_GDP",
        "Imports of goods and services (% of GDP)": "Imports_GDP",
        "Agriculture, forestry, and fishing, value added (% of GDP)": "Agriculture_GDP",
        "Industry (including construction), value added (% of GDP)": "Industry_GDP",
        "Manufacturing, value added (% of GDP)": "Manufacturing_GDP",
        "Services, value added (% of GDP)": "Services_GDP",
        "Foreign direct investment, net inflows (% of GDP)": "FDI_Inflows_GDP",
        "Access to electricity (% of population)": "Access_Electricity",
        "Mobile cellular subscriptions (per 100 people)": "Mobile_Subscriptions",
        "Individuals using the Internet (% of population)": "Internet_Usage",
        "Battle-related deaths (number of people)": "Battle_Deaths",
        "Oil rents (% of GDP)": "Oil_Rents",
        "Government expenditure on education, total (% of GDP)": "Education_Expenditure",
        "Current health expenditure (% of GDP)": "Health_Expenditure"
    }
    
    # Filter dataset for selected indicators
    print("Filtering for key indicators...")
    df_filtered = df[df["Indicator Name"].isin(indicators_map.keys())].copy()
    
    df_filtered["Indicator_Code_Simple"] = df_filtered["Indicator Name"].map(indicators_map)
    df_filtered = df_filtered.drop(columns=["Indicator Name", "Indicator Code"])
    
    # Transpose to wide format
    id_vars = ["Indicator_Code_Simple"]
    df_melted = df_filtered.melt(id_vars=id_vars, var_name="Year", value_name="Value")
    df_pivot = df_melted.pivot(index="Year", columns="Indicator_Code_Simple", values="Value")
    
    # Clean Index and convert to numeric
    df_pivot.index = df_pivot.index.astype(int)
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.apply(pd.to_numeric, errors='coerce')
    
    # Handle Missing Values
    # 1. For technology indicators that didn't exist in the past, fill initial NaNs with 0.
    tech_indicators = ["Internet_Usage", "Mobile_Subscriptions"]
    for col in tech_indicators:
        if col in df_pivot.columns:
            df_pivot[col] = df_pivot[col].fillna(0)

    # 2. For other indicators, use linear interpolation (forward only to respect time).
    # We avoid 'limit_direction=both' to prevent future data leaking into the past.
    print("Handling missing values (Interpolation)...")
    df_clean = df_pivot.interpolate(method='linear', limit_direction='forward')
    
    # 3. Drop rows only if ALL values are missing (unlikely, but good cleanup)
    # We removed the strict 'dropna(how='any')' because it was deleting the 1990s data 
    # (likely due to missing Education/Health data in early years).
    df_clean = df_clean.dropna(how='all') 

    print(f"Saving processed data to {processed_path}...")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_clean.to_csv(processed_path)
    
    print("Data cleaning complete.")
    print(f"Shape: {df_clean.shape}")
    print(f"Columns: {df_clean.columns.tolist()}")

if __name__ == "__main__":
    clean_data()
