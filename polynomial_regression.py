import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

np.random.seed(42)

# -----------------------------
# Data Loading
# -----------------------------

def load_data(filepath='train.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)


def preprocess_data(df):
    df = df.copy()
    df = df.drop('Id', axis=1, errors='ignore')
    
    y = df['SalePrice'].copy()
    X = df.drop('SalePrice', axis=1)
    
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna('None', inplace=True)
    
    X = pd.get_dummies(X, drop_first=True)
    return X, y

# -----------------------------
# Feature Engineering
# -----------------------------

def select_top_features(X_train, y_train, X_test, k=15):
    correlations = {}
    for col in X_train.columns:
        corr = np.corrcoef(X_train[col], y_train)[0, 1]
        correlations[col] = abs(corr)
    
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:k]
    selected_cols = [feat for feat, _ in top_features]
    
    print(f"\nTop {k} features by correlation:")
    for i, (feat, corr) in enumerate(top_features, 1):
        print(f"  {i}. {feat[:30]}: {corr:.4f}")
    
    return X_train[selected_cols], X_test[selected_cols], selected_cols


def build_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))
    X_poly = np.hstack([X_poly, X])
    
    if degree > 1:
        for d in range(2, degree + 1):
            X_poly = np.hstack([X_poly, X ** d])
    
    return X_poly

# -----------------------------
# Model Training
# -----------------------------

def fit_polynomial_regression(X, y, degree, alpha=1.0):
    X_poly = build_polynomial_features(X, degree)
    print(f"Polynomial features: {X_poly.shape[1]}")
    
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    return model


def predict_polynomial(model, X, degree):
    X_poly = build_polynomial_features(X, degree)
    return model.predict(X_poly)


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    print("\nHyperparameter tuning:")
    print("-" * 60)
    
    degrees = [1, 2, 3]
    alphas = [0.1, 1.0, 10.0, 100.0]
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for degree in degrees:
        for alpha in alphas:
            model = fit_polynomial_regression(X_train, y_train, degree, alpha)
            y_val_pred = predict_polynomial(model, X_val, degree)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            results.append({'degree': degree, 'alpha': alpha, 'val_rmse': val_rmse})
            print(f"Degree={degree}, Alpha={alpha:6.1f} -> Val RMSE: ${val_rmse:,.2f}")
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {'degree': degree, 'alpha': alpha}
    
    print("-" * 60)
    print(f"Best: Degree={best_params['degree']}, Alpha={best_params['alpha']}")
    print(f"Best Val RMSE: ${best_score:,.2f}")
    
    pd.DataFrame(results).to_csv('polynomial_regression_tuning.csv', index=False)
    return best_params['degree'], best_params['alpha']

# -----------------------------
# Evaluation
# -----------------------------

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}

# -----------------------------
# Visualization
# -----------------------------

def plot_results(y_true, y_pred, degree):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.grid(True)
    
    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_results.png', dpi=300)
    plt.show()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("="*60)
    print("POLYNOMIAL REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    df = load_data('train.csv')
    print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    X, y = preprocess_data(df)
    print(f"Features after preprocessing: {X.shape[1]}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    X_train_sel, X_val_sel, selected_features = select_top_features(X_train, y_train, X_val, k=15)
    X_test_sel = X_test[selected_features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    best_degree, best_alpha = tune_hyperparameters(X_train_scaled, y_train.values, X_val_scaled, y_val.values)
    
    print(f"\nTraining final model (degree={best_degree}, alpha={best_alpha})...")
    model = fit_polynomial_regression(X_train_scaled, y_train.values, best_degree, best_alpha)
    
    y_train_pred = predict_polynomial(model, X_train_scaled, best_degree)
    y_test_pred = predict_polynomial(model, X_test_scaled, best_degree)
    
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)
    
    print("\nResults:")
    print(f"  Train - RMSE: ${train_metrics['rmse']:,.2f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Test  - RMSE: ${test_metrics['rmse']:,.2f}, R²: {test_metrics['r2']:.4f}")
    
    plot_results(y_test.values, y_test_pred, best_degree)
    
    results_df = pd.DataFrame({
        'Metric': ['Degree', 'Alpha', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2'],
        'Value': [best_degree, best_alpha, train_metrics['rmse'], test_metrics['rmse'], 
                  train_metrics['r2'], test_metrics['r2']]
    })
    results_df.to_csv('polynomial_regression_metrics.csv', index=False)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()