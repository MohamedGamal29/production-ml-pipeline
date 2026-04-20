import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model():
    print(" Evaluating model performance ...")



    # 1. Load data and split
    df = pd.read_csv('data/clean_data.csv')
    x = df[['Square_Feet', 'Bedrooms', 'Age_Years', 'price_per_sqft']]
    y = df['Price']
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # 2, load saved model
    model = joblib.load('models/rf_model.pk1')



    # 3. predict and calculate metrics
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)


    print(f"Evaluation complete ")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R-Squared Score (Accuracy): {r2 * 100:.2f}%")


if __name__ == "__main__":
    evaluate_model()