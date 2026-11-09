# 🚗 Task 3: Car Price Prediction with Machine Learning

## 📋 Project Overview
This project builds an automated used car valuation system using supervised machine learning regression algorithms. It predicts fair resale price based on vehicle attributes such as showroom price (`Present_Price`), vehicle age, mileage (`Kms_Driven`), fuel type, transmission type, and seller type.

## 📁 Folder Structure
```
Task_3_Car_Price_Prediction/
├── dataset/
│   └── car data.csv                             # Vehicle resale listings dataset
├── outputs/
│   ├── figures/                                 # Visualizations & regression diagnostics
│   │   ├── 01_price_distribution.png
│   │   ├── 02_present_vs_selling_price.png
│   │   ├── 03_categorical_price_impact.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_model_performance_comparison.png
│   │   ├── 06_actual_vs_predicted_prices.png
│   │   ├── 07_residual_analysis.png
│   │   └── 08_feature_importances.png
│   └── car_price_model.joblib                   # Serialized pipeline (preprocessor + model)
├── car_price_prediction.py                      # Standalone Python training pipeline
├── car_price_prediction.ipynb                   # Interactive Jupyter Notebook
└── README.md                                    # Task documentation
```

## 🔬 Methodology & Modeling
1. **Feature Engineering:**
   - Derived `Car_Age = Current_Year - Year` to capture depreciation kinetics.
   - Built an end-to-end `ColumnTransformer` applying `StandardScaler` to numerical inputs and `OneHotEncoder` to categorical factors (`Fuel_Type`, `Seller_Type`, `Transmission`).
2. **Algorithm Benchmarking:**
   - Evaluated 5 regression algorithms across 5-Fold Cross-Validation.
   - **Gradient Boosting Regressor** emerged as the top-performing model, achieving **$R^2 = 0.9724$** and an **MAE of just 0.52 Lakhs INR** on unseen test data.
3. **Key Price Drivers:**
   - `Present_Price` (original showroom cost) accounts for >80% of predictive power.
   - Vehicle age and transmission type (automatic commands higher premium) are secondary determinants.

## 📊 Benchmark Results
| Model | 5-Fold CV $R^2$ | Test $R^2$ | MAE (Lakhs) | RMSE (Lakhs) |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.9182** | **0.9724** | **0.520** | **0.798** |
| **Random Forest Regressor** | 0.9019 | 0.9574 | 0.648 | 0.991 |
| **Ridge Regression** | 0.8048 | 0.8492 | 1.214 | 1.864 |
| **Linear Regression** | 0.8009 | 0.8490 | 1.216 | 1.865 |
| **Lasso Regression** | 0.8018 | 0.8477 | 1.225 | 1.873 |

## 🚀 How to Run
```bash
# Execute standalone modeling script
python car_price_prediction.py

# Or launch Jupyter Notebook
jupyter notebook car_price_prediction.ipynb
```
