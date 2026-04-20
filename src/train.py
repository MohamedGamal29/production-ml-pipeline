import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def train_model():
    print(" Loading data and training model ...")


    # 1. Load data
    df = pd.read_csv('data/clean_data.csv')
    x = df[['Square_Feet', 'Bedrooms', 'Age_Years', 'price_per_sqft']]
    y = df['Price']


    # 2. split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # 3. train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # 4. save the trained model 
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_model.pk1')


    print(" Model Traind and saved successfully to 'model/rf_model.pk1'!")

if __name__ == "__main__":
    train_model()

