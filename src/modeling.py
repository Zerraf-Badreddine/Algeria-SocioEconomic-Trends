
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, plot_importance
import warnings
import os

warnings.filterwarnings('ignore')

# Settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGURES_PATH = os.path.join('reports', 'figures')
os.makedirs(FIGURES_PATH, exist_ok=True)

DATA_PATH = os.path.join('data', 'processed', 'algeria_featured.csv')

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Year'] = pd.to_datetime(df['Year'])
    df = df.set_index('Year')
    return df

def run_baseline_models(X_train, X_test, y_train, y_test):
    print("\n--- Baseline Models (Linear/Lasso/Ridge) ---")
    
    # Scale data for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(alpha=0.1),
        "Ridge": Ridge(alpha=1.0)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        metrics = f"RMSE: {rmse:.4f}"
        print(f"{name}: {metrics}")
        results[name] = rmse
        predictions[name] = preds
        
        # Save coefficients for Ridge (best linear proxy usually)
        if name == "Ridge":
            coefs = pd.Series(model.coef_, index=X_train.columns).sort_values()
            plt.figure(figsize=(10, 8))
            coefs.plot(kind='barh')
            plt.title('Feature Coefficients (Ridge Regression)')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_PATH, 'coefficients_ridge.png'))
            plt.close()

    return results, predictions

def run_var_analysis(df):
    print("\n--- VAR Analysis --")
    var_cols = ['GDP_Growth', 'Oil_Rents', 'Inflation']
    data = df[var_cols].dropna()
    
    model = VAR(data)
    lag_selection = model.select_order(maxlags=4)
    if not lag_selection.aic:
        optimal_lag = 1
    else:
        optimal_lag = lag_selection.aic
    print(f"Optimal Lag: {optimal_lag}")
    
    var_model = model.fit(optimal_lag)
    irf = var_model.irf(10)
    
    # Plot IRF: Oil Rents -> GDP Growth
    plt.figure(figsize=(10, 6))
    irf.plot(orth=False, impulse='Oil_Rents', response='GDP_Growth')
    plt.title('Impulse Response: Oil Rents Shock -> GDP Growth')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'irf_oil_gdp.png'))
    plt.close()
    
    return var_model

def run_xgboost_forecasting(X_train, X_test, y_train, y_test):
    print("\n--- XGBoost Forecasting ---")
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"XGBoost RMSE: {rmse:.4f}")
    
    # Feature Importance
    plt.figure(figsize=(10, 8))
    plot_importance(xgb, max_num_features=12, height=0.6)
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'feature_importance_xgboost.png'))
    plt.close()
    
    return rmse, y_pred

def plot_forecasts(y_test, baseline_preds, xgb_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2, marker='o')
    
    # Plot Lasso (Representative Baseline)
    if 'Lasso' in baseline_preds:
        plt.plot(y_test.index, baseline_preds['Lasso'], label='Lasso (Baseline)', linestyle='--')
        
    # Plot XGBoost
    plt.plot(y_test.index, xgb_pred, label='XGBoost', linestyle='-.', color='red')
    
    plt.title('Forecast Comparison: Actual vs Baseline vs Advanced')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'forecast_comparison.png'))
    plt.close()

def main():
    df = load_data()
    
    # Prepare Data
    target = 'GDP_Growth'
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    
    split_date = '2015-01-01'
    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]
    
    # 1. Baseline
    results, baseline_preds = run_baseline_models(X_train, X_test, y_train, y_test)
    
    # 2. VAR
    try:
        run_var_analysis(df)
    except Exception as e:
        print(f"VAR failed: {e}")
        
    # 3. XGBoost
    xgb_rmse, xgb_pred = run_xgboost_forecasting(X_train, X_test, y_train, y_test)
    results['XGBoost'] = xgb_rmse
    
    # 4. Compare
    print("\n--- Final Results (RMSE) ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    # 5. Plot Comparison
    plot_forecasts(y_test, baseline_preds, xgb_pred)
    
    # Performance Bar Chart
    plt.figure(figsize=(8, 5))
    pd.Series(results).sort_values().plot(kind='bar', color='skyblue')
    plt.title('Model Performance Benchmark (RMSE)')
    plt.ylabel('Root Mean Squared Error (Lower is Better)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'model_benchmark.png'))
    plt.close()

if __name__ == "__main__":
    main()
