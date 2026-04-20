# Production ML Pipeline: House Price Prediction

This repository demonstrates a modular, production-ready Machine Learning pipeline. Instead of a single messy Jupyter Notebook, the codebase is structured into scalable modules for preprocessing, training, and evaluation.

##  Project Structure
- `src/preprocessing.py`: Cleans raw data, handles missing values, and performs feature engineering.
- `src/train.py`: Trains a Random Forest model and serializes it using `joblib`.
- `src/evaluate.py`: Loads the serialized model and evaluates it using MAE and R2-Score.

##  How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data pipeline: `python src/preprocessing.py`
3. Train model: `python src/train.py`
4. Evaluate: `python src/evaluate.py`