
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import os

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
DATA_PATH = os.path.join('data', 'processed', 'algeria_cleaned.csv')
OUTPUT_PATH = os.path.join('data', 'processed', 'algeria_featured.csv')
FIGURES_PATH = os.path.join('reports', 'figures')

os.makedirs(FIGURES_PATH, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df = df.set_index('Year')

    # --- Data Cleaning ---
    print("Performing imputation...")
    # 1. Linear Interpolation
    df_imputed = df.interpolate(method='linear', limit_direction='both')
    
    # 2. KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    df_values = imputer.fit_transform(df_imputed)
    df_imputed = pd.DataFrame(df_values, columns=df_imputed.columns, index=df_imputed.index)

    # Save Missing Values Heatmap (Cleaned)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_imputed.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap (After Cleaning)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'missing_values_cleaned.png'))
    plt.close()

    # --- Feature Construction ---
    print("Constructing features...")
    df_imputed['Trade_Balance_GDP'] = df_imputed['Exports_GDP'] - df_imputed['Imports_GDP']
    df_imputed['Trade_Openness'] = df_imputed['Exports_GDP'] + df_imputed['Imports_GDP']
    
    if 'Industry_GDP' in df_imputed.columns and 'Manufacturing_GDP' in df_imputed.columns:
        df_imputed['Industry_Non_Mfg_GDP'] = df_imputed['Industry_GDP'] - df_imputed['Manufacturing_GDP']

    # Lags and Rolling
    lag_cols = ['GDP_Growth', 'Inflation', 'Oil_Rents']
    for col in lag_cols:
        if col in df_imputed.columns:
            df_imputed[f'{col}_Lag1'] = df_imputed[col].shift(1)

    roll_cols = ['GDP_Growth', 'Inflation']
    for col in roll_cols:
        if col in df_imputed.columns:
            df_imputed[f'{col}_Roll3'] = df_imputed[col].rolling(window=3).mean()

    # Drop NaNs from temporal features
    df_imputed = df_imputed.dropna()

    # --- Transformations ---
    print("Applying transformations...")
    skewed_features = ['GDP_USD', 'Population', 'CO2_Emissions']
    for col in skewed_features:
        if col in df_imputed.columns:
            df_imputed[f'Log_{col}'] = np.log1p(df_imputed[col])

    # Save Distribution Plot
    if 'GDP_USD' in df_imputed.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df_imputed['GDP_USD'], kde=True, ax=axes[0]).set_title('Original GDP_USD')
        sns.histplot(df_imputed['Log_GDP_USD'], kde=True, ax=axes[1]).set_title('Log Transformed GDP_USD')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'log_transformation_gdp.png'))
        plt.close()

    # --- Feature Selection / Correlation ---
    print("Generating correlation matrix...")
    corr_matrix = df_imputed.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'feature_correlation_matrix.png'))
    plt.close()

    # --- Export ---
    print(f"Saving processed data to {OUTPUT_PATH}...")
    df_imputed.to_csv(OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
