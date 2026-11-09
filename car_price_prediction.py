"""
Task 3: Car Price Prediction with Machine Learning
CodeAlpha Data Science Internship
Author: CodeAlpha Intern
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.autolayout'] = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, "dataset", "car data.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_engineer_features(path=DATASET_PATH):
    print("=" * 65)
    print("STEP 1: Data Ingestion & Feature Engineering")
    print("=" * 65)
    
    df = pd.read_csv(path)
    print(f"Raw dataset shape: {df.shape}")
    print("\nFirst 5 records:")
    print(df.head())
    
    # Feature Engineering: Car Age (Reference current year: 2024)
    df['Car_Age'] = 2024 - df['Year']
    
    # Extract vehicle brand/make from Car_Name
    df['Brand'] = df['Car_Name'].apply(lambda x: str(x).split()[0].lower())
    
    # Check nulls
    print("\nMissing values:\n", df.isnull().sum())
    print("\nCategorical summary:")
    for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']:
        print(f"{col}: {dict(df[col].value_counts())}")
        
    return df


def perform_eda(df, output_dir=FIGURES_DIR):
    print("\n" + "=" * 65)
    print("STEP 2: Exploratory Data Analysis & Visualizations")
    print("=" * 65)
    
    # 1. Price Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['Selling_Price'], kde=True, color='#1f77b4', ax=axes[0])
    axes[0].set_title("Resale Selling Price Distribution (in Lakhs)", fontweight='bold')
    axes[0].set_xlabel("Selling Price (Lakhs INR)")
    
    sns.boxplot(x=df['Selling_Price'], color='#aec7e8', ax=axes[1])
    axes[1].set_title("Selling Price Boxplot (Outlier Check)", fontweight='bold')
    axes[1].set_xlabel("Selling Price (Lakhs INR)")
    
    p1 = os.path.join(output_dir, "01_price_distribution.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")
    
    # 2. Present Price vs Selling Price Scatter
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df, x='Present_Price', y='Selling_Price',
                scatter_kws={'alpha': 0.7, 'color': '#2ca02c'}, line_kws={'color': '#d62728', 'linewidth': 2})
    plt.title("Resale Price vs. Showroom (Present) Price", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Present Price (Lakhs INR)", fontweight='bold')
    plt.ylabel("Selling Price (Lakhs INR)", fontweight='bold')
    p2 = os.path.join(output_dir, "02_present_vs_selling_price.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")
    
    # 3. Categorical Impact on Selling Price
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, ax=axes[0], palette='Set2')
    axes[0].set_title("Price by Fuel Type", fontweight='bold')
    axes[0].set_ylabel("Selling Price (Lakhs)")
    
    sns.boxplot(x='Transmission', y='Selling_Price', data=df, ax=axes[1], palette='Set2')
    axes[1].set_title("Price by Transmission", fontweight='bold')
    axes[1].set_ylabel("")
    
    sns.boxplot(x='Seller_Type', y='Selling_Price', data=df, ax=axes[2], palette='Set2')
    axes[2].set_title("Price by Seller Type", fontweight='bold')
    axes[2].set_ylabel("")
    
    plt.suptitle("Market Segment Resale Price Distributions", fontsize=15, fontweight='bold')
    p3 = os.path.join(output_dir, "03_categorical_price_impact.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt=".3f", linewidths=1.0)
    plt.title("Numeric Features Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
    p4 = os.path.join(output_dir, "04_correlation_heatmap.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")


def train_and_evaluate_models(df, output_dir=FIGURES_DIR):
    print("\n" + "=" * 65)
    print("STEP 3: Regression Modeling, Evaluation & Comparison")
    print("=" * 65)
    
    # Feature Selection
    numeric_features = ['Present_Price', 'Kms_Driven', 'Car_Age', 'Owner']
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    target = 'Selling_Price'
    
    X = df[numeric_features + categorical_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    candidate_regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.05),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, regressor in candidate_regressors.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        # 5-fold cross validation R2
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "CV R² (Mean)": cv_r2.mean(),
            "CV R² (Std)": cv_r2.std(),
            "Test R²": r2,
            "MAE": mae,
            "RMSE": rmse
        })
        trained_pipelines[name] = pipe
        
    results_df = pd.DataFrame(results).sort_values(by="Test R²", ascending=False)
    print("\nModel Benchmark Performance Table:")
    print(results_df.to_string(index=False))
    
    # 5. Model Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(x="Test R²", y="Model", data=results_df, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Test R² Score Comparison", fontweight='bold')
    axes[0].set_xlim(0.7, 1.0)
    for i, v in enumerate(results_df["Test R²"]):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va='center', fontweight='bold')
        
    sns.barplot(x="RMSE", y="Model", data=results_df, ax=axes[1], palette="Reds_r")
    axes[1].set_title("Root Mean Squared Error (Lower is Better)", fontweight='bold')
    for i, v in enumerate(results_df["RMSE"]):
        axes[1].text(v + 0.02, i, f"{v:.3f}", va='center', fontweight='bold')
        
    p5 = os.path.join(output_dir, "05_model_performance_comparison.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved: {p5}")
    
    # Best Model Deep Dive
    best_name = results_df.iloc[0]["Model"]
    best_pipe = trained_pipelines[best_name]
    print(f"\nOptimal Model Selected: {best_name}")
    
    y_pred_best = best_pipe.predict(X_test)
    residuals = y_test - y_pred_best
    
    # 6. Actual vs Predicted Prices Plot
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_pred_best, color='#1f77b4', edgecolors='k', alpha=0.8, s=60)
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label="Perfect Fit (y = x)")
    plt.title(f"Actual vs Predicted Car Resale Prices ({best_name})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Actual Resale Price (Lakhs INR)", fontweight='bold')
    plt.ylabel("Predicted Resale Price (Lakhs INR)", fontweight='bold')
    plt.legend(frameon=True)
    p6 = os.path.join(output_dir, "06_actual_vs_predicted_prices.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}")
    
    # 7. Residual Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals, kde=True, color='#2ca02c', ax=axes[0])
    axes[0].set_title("Residuals Distribution (Error Normality)", fontweight='bold')
    axes[0].set_xlabel("Residual Error (Actual - Predicted)")
    
    axes[1].scatter(y_pred_best, residuals, color='#ff7f0e', edgecolors='k', alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Residuals vs. Predicted Values (Homoscedasticity)", fontweight='bold')
    axes[1].set_xlabel("Predicted Selling Price")
    axes[1].set_ylabel("Residuals")
    
    p7 = os.path.join(output_dir, "07_residual_analysis.png")
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"Saved: {p7}")
    
    # 8. Feature Importances (Tree model)
    rf_pipe = trained_pipelines["Random Forest Regressor"]
    onehot_encoder = rf_pipe.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = list(onehot_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + encoded_cat_names
    
    importances = rf_pipe.named_steps['regressor'].feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=True)
    
    plt.figure(figsize=(9, 6))
    feat_imp.plot(kind='barh', color='#17becf')
    plt.title("Car Price Determinants: Feature Importance", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Importance Weight", fontweight='bold')
    p8 = os.path.join(output_dir, "08_feature_importances.png")
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"Saved: {p8}")
    
    # Model Persistence
    save_package = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "feature_names": numeric_features + categorical_features,
        "metrics": results_df.iloc[0].to_dict()
    }
    model_save_path = os.path.join(OUTPUT_DIR, "car_price_model.joblib")
    joblib.dump(save_package, model_save_path)
    print(f"\nPipeline successfully saved to: {model_save_path}")
    
    return save_package, results_df


def predict_car_price(present_price, kms_driven, car_age, owner, fuel_type, seller_type, transmission,
                      model_path=os.path.join(OUTPUT_DIR, "car_price_model.joblib")):
    package = joblib.load(model_path)
    pipe = package["pipeline"]
    
    input_df = pd.DataFrame([{
        'Present_Price': float(present_price),
        'Kms_Driven': int(kms_driven),
        'Car_Age': int(car_age),
        'Owner': int(owner),
        'Fuel_Type': str(fuel_type),
        'Seller_Type': str(seller_type),
        'Transmission': str(transmission)
    }])
    
    predicted_price = pipe.predict(input_df)[0]
    return max(0.0, float(predicted_price))


if __name__ == "__main__":
    df = load_and_engineer_features()
    perform_eda(df)
    package, results_df = train_and_evaluate_models(df)
    
    print("\n" + "=" * 65)
    print("STEP 4: Testing Model Inference on Sample Vehicles")
    print("=" * 65)
    test_cars = [
        # (Present_Price, Kms, Age, Owner, Fuel, Seller, Trans)
        {"name": "Maruti Swift (4 yrs, Petrol)", "params": (6.5, 35000, 4, 0, 'Petrol', 'Dealer', 'Manual')},
        {"name": "Hyundai Creta (3 yrs, Diesel)", "params": (13.5, 40000, 3, 0, 'Diesel', 'Dealer', 'Automatic')},
        {"name": "Toyota Fortuner (6 yrs, Diesel)", "params": (32.0, 75000, 6, 1, 'Diesel', 'Individual', 'Automatic')}
    ]
    for car in test_cars:
        est_price = predict_car_price(*car["params"])
        print(f"Vehicle: {car['name']}")
        print(f"Inputs: {car['params']} => Estimated Fair Market Value: {est_price:.2f} Lakhs INR\n")
        
    print("Task 3 execution completed successfully!")
