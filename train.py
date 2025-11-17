import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import joblib  # <-- 1. IMPORTED JOBLIB

# --- 1. Configuration ---
DATA_FILE_PATH = os.path.join("data", "output", "final_groundwater_data.csv")
MODEL_OUTPUT_DIR = "data/model_output"

# Create directory for model outputs if it doesn't exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_data(file_path):
    """
    Loads the dataset, assigns column names, handles data types, and performs initial cleaning.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return None

    print("Loading and preparing data...")

    # *** FIX: DEFINE THE COLUMN NAMES ***
    # This list must match the order of columns in your CSV file.
    column_names = [
        'STATE_UT', 'DISTRICT', 'BLOCK', 'VILLAGE',
        'LATITUDE', 'LONGITUDE', 'DATE', 'DTWL',
        'SOURCE_FILE', 'METHOD', 'INDEX', 'EXTRA' # Naming the extra columns from the extraction
    ]

    # *** FIX: USE THE 'names' PARAMETER TO ASSIGN THE COLUMN HEADERS ***
    df = pd.read_csv(file_path, names=column_names)

    # --- Data Cleaning and Type Conversion ---
    # Now that the 'DATE' column has a name, this line will work
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce', format='%d-%m-%y')

    # Define columns to convert to numeric, coercing errors to NaN
    numeric_cols = ['LATITUDE', 'LONGITUDE', 'DTWL']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential data (date, coordinates, target) is missing
    df.dropna(subset=['DATE', 'LATITUDE', 'LONGITUDE', 'DTWL'], inplace=True)

    print(f"Data loaded successfully with {df.shape[0]} valid records.")
    return df


# --- 3. Feature Engineering ---
def create_features(df):
    """
    Creates new features from existing data to improve model performance.
    """
    print("Creating features for the model...")
    df_featured = df.copy()

    # --- Time-based Features ---
    df_featured['YEAR'] = df_featured['DATE'].dt.year
    df_featured['MONTH'] = df_featured['DATE'].dt.month
    df_featured['DAY_OF_YEAR'] = df_featured['DATE'].dt.dayofyear

    # --- Geospatial Features ---
    df_featured['LAT_LON_PRODUCT'] = df_featured['LATITUDE'] * df_featured['LONGITUDE']

    print("Features created: YEAR, MONTH, DAY_OF_YEAR, LAT_LON_PRODUCT")
    return df_featured


# --- 4. Model Training and Evaluation ---
def train_and_evaluate_model(df):
    """
    Selects features, splits data, trains a RandomForest model, and evaluates it.
    """
    target = 'DTWL'
    features = [
        'LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH',
        'DAY_OF_YEAR', 'LAT_LON_PRODUCT'
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    print("Training the RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} meters")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} meters")
    print(f"Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")

    print("\n--- Feature Importance ---")
    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    print(importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
    plt.title('Feature Importance for Groundwater Level Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "feature_importance.png"))
    print("\nFeature importance plot saved to 'data/model_output/feature_importance.png'")

    # --- 2. ADDED CODE TO SAVE THE MODEL ---
    model_filename = os.path.join(MODEL_OUTPUT_DIR, "groundwater_model.joblib")
    joblib.dump(model, model_filename)
    print(f"\n--- Model Saved ---")
    print(f"Model has been saved to: '{model_filename}'")
    # ------------------------------------

    return model

# --- 5. Example Prediction ---
def predict_new_data(model):
    """
    Demonstrates how to use the trained model to predict on new, unseen data.
    """
    print("\n--- Example Prediction ---")
    new_data = pd.DataFrame({
        'LATITUDE': [12.5, 11.6],
        'LONGITUDE': [92.8, 92.7],
        'DATE': pd.to_datetime(['2025-05-15', '2025-11-20'])
    })

    new_data_featured = create_features(new_data)
    prediction_features = new_data_featured[['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY_OF_YEAR', 'LAT_LON_PRODUCT']]
    
    predictions = model.predict(prediction_features)

    for i, row in new_data.iterrows():
        print(f"Predicted DTWL for Lat={row['LATITUDE']}, Lon={row['LONGITUDE']} on {row['DATE'].date()}: {predictions[i]:.2f} meters")


def investigate_dropped_rows(file_path):
    """
    Loads the raw data and shows examples of rows that are dropped during cleaning.
    """
    print("\n--- Investigating Dropped Rows ---")
    column_names = [
        'STATE_UT', 'DISTRICT', 'BLOCK', 'VILLAGE',
        'LATITUDE', 'LONGITUDE', 'DATE', 'DTWL',
        'SOURCE_FILE', 'METHOD', 'INDEX', 'EXTRA'
    ]
    
    # Load the raw data without any cleaning
    df_raw = pd.read_csv(file_path, names=column_names)
    
    # Create a copy to perform cleaning on
    df_clean = df_raw.copy()

    # Perform the same conversions
    df_clean['DATE'] = pd.to_datetime(df_clean['DATE'], errors='coerce', format='%d-%m-%y')
    for col in ['LATITUDE', 'LONGITUDE', 'DTWL']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Find the rows that have at least one null value in the key columns
    dropped_mask = df_clean[['DATE', 'LATITUDE', 'LONGITUDE', 'DTWL']].isnull().any(axis=1)
    
    # Get the original data for the dropped rows
    dropped_rows = df_raw[dropped_mask]
    
    print(f"Found {len(dropped_rows)} rows that would be dropped due to missing/invalid data.")
    print("Here are the first 10 examples of dropped rows (showing key columns):")
    
    # Display the problematic columns from the original data
    print(dropped_rows[['LATITUDE', 'LONGITUDE', 'DATE', 'DTWL']].head(10))


# --- Main Execution ---
if __name__ == "__main__":
    df = load_and_prepare_data(DATA_FILE_PATH)
    investigate_dropped_rows(DATA_FILE_PATH)

    if df is not None:
        df_featured = create_features(df)
        trained_model = train_and_evaluate_model(df_featured)
        if trained_model:
            predict_new_data(trained_model)