import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def main():
    print("=== AI HOUSE PRICE PREDICTION MODEL TRAINING ===")
    data_path = os.path.join("data", "house_prices.csv")
    model_path = os.path.join("model", "house_price_model.pkl")
    
    df = pd.read_csv(data_path)
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)

    feature_cols = ["Area_SqFt", "Bedrooms", "Bathrooms", "Floors", "Parking", "House_Age", "Location_Tier"]
    X = df[feature_cols]
    y = df["Price_INR"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")

    joblib.dump(model, model_path)
    print("✓ Model saved successfully.")

if __name__ == "__main__":
    main()