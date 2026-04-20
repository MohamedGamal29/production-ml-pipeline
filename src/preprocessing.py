import pandas as pd 
import numpy as np
import os


def prepare_data():
    print("Starting data preparation...")


    # 1. Create dummy row data
    np.random.seed(42)
    data = {
        'Square_Feet': np.random.randint(800, 4000, 200),
        'Bedrooms': np.random.randint(1, 6, 200),
        'Age_Years': np.random.randint(0, 50, 200),
        'Price': np.random.randint(100000, 600000, 200)
    }
    df = pd.DataFrame(data)


    # Introduce some missing values to show cleaning skills
    df.loc[10:15, 'Bedrooms']= np.nan


    # 2. Data Cleaning (handling missing values)
    df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())


    # 3. Feature engineering
    df['price_per_sqft'] = df['Price'] / df['Square_Feet']


    # 4. Save clean data
    df.to_csv('data/clean_data.csv', index=False)
    print("Clean data saved succesfully to 'data/clean_data.csv'!")


if __name__== "__main__":
    prepare_data()