from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

def load_model(model_name):
    # Define the absolute path to the models directory
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    return joblib.load(os.path.join(models_dir, f"{model_name}.joblib"))

logistic_regression_model = load_model("logistic_regression")
random_forest_model = load_model("random_forest")
gradient_boosting_model = load_model("gradient_boosting")
support_vector_machine_model = load_model("support_vector_machine")
isolation_forest_model = load_model("isolation_forest")
neural_network_model = load_model("neural_network")
ensemble_learning_model = load_model("ensemble_learning")

class Transaction(BaseModel):
  features: list

@app.post("/predict/logistic_regression")
def predict_logistic_regression(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = logistic_regression_model.predict(features)
  return {"prediction": int(prediction[0])}

@app.post("/predict/random_forest")
def predict_random_forest(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = random_forest_model.predict(features)
  return {"prediction": int(prediction[0])}

@app.post("/predict/gradient_boosting")
def predict_gradient_boosting(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = gradient_boosting_model.predict(features)
  return {"prediction": int(prediction[0])}

@app.post("/predict/svm")
def predict_svm(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = support_vector_machine_model.predict(features)
  return {"prediction": int(prediction[0])}

@app.post("/predict/isolation_forest")
def predict_isolation_forest(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = isolation_forest_model.predict(features)
  return {"prediction": int(prediction[0])}

@app.post("/predict/neural_network")
def predict_neural_network(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = neural_network_model.predict(features)
  return {"prediction": int(prediction[0])}

@app.post("/predict/ensemble")
def predict_ensemble(transaction: Transaction):
  features = pd.DataFrame([transaction.features])
  prediction = ensemble_learning_model.predict(features)
  return {"prediction": int(prediction[0])}