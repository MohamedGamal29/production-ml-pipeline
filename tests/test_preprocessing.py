import os 
import sys
import pandas as pd
from src.preprocessing import prepare_data

def test_data_cleaning_and_feature_engineering():
    # Run the data preparation function
    prepare_data()


    # Check if the cleaned data file exists
    assert os.path.exists('data/clean_data.csv'), "Cleaned data file not found!"


    # Load the cleaned data and check for missing values
    df = pd.read_csv('data/clean_data.csv')
    tota_missing = df.isnull().sum().sum()
    assert tota_missing == 0, f"found {tota_missing} missing values Cleaning failed"

    # Check if the new feature 'price_per_sqft' is created correctly
    assert 'price_per_sqft' in df.columns, "Feature engineering failed: 'price_per_sqft' not found"
    print("Data cleaning test passed successfully!")
