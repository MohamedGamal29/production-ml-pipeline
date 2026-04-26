import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np




def run_cross_validation():
    print(" Loading data for Cross-validation...")


    # 1. load the clean data
    df = pd.read_csv("data/clean_data.csv")
    x = df[['Square_Feet', 'Bedrooms', 'Age_Years', 'price_per_sqft']]
    y = df['Price']



    # 2. Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)



    # 3. K-fold Cross validation (k=5)
    print(" Runing 5-Fold Cross Validation ... (taking 5 different exams)")

    scores = cross_val_score(model, x, y, cv=5, scoring='r2')



    # 4. print the magic results
    print (" \n Results of the 5 exams :")
    for i, score in enumerate(scores):
        print(f"Fold {i+1}: Accuracy ={score*100:.2f}%")


    print(f"\n The True average Accurracy : {np.mean(scores)*100:.2f}%")
    print(f" Variance (Stability): +\- {np.std(scores)*100:.2f}% If this is high, the model is unstable ")




if __name__ == "__main__":
    run_cross_validation() 