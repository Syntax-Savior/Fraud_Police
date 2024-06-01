from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import os
import joblib

models_dir = os.path.join(os.path.dirname(file))
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def save_model(model, model_name):
    joblib.dump(model, os.path.join(models_dir, f"{model_name}.joblib"))

# DEFININF FUNCTIONS TO TRAIN VARIOUS MODELS.
# LOGISTIC REGRESSION
def train_logistic_regression(X_train, y_train):
  param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
  }
  grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train, y_train)

  lr_model = grid.best_estimator_
  save_model(lr_model, "logistic_regression")

  return lr_model

# RANDOM FOREST
def train_random_forest(X_train, y_train):
  param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
  }
  grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train, y_train)

  rf_model = grid.best_estimator_
  save_model(rf_model, "random_forest")

  return rf_model

# GRADIENT BOOSTING
def train_gradient_boosting(X_train, y_train):
  param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
  }
  grid = GridSearchCV(GradientBoostingClassifier(), param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train, y_train)

  gb_model = grid.best_estimator_
  save_model(gb_model, "gradient_boosting")

  return gb_model

# SVM
def train_svm(X_train, y_train):
  param_grid = {
    'C': [0.1, 1, 10, 100],
    'Kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
  }
  grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train, y_train)

  svm_model = grid.best_estimator_
  save_model(svm_model, "support_vector_machine")

  return svm_model

# ISOLATION FOREST
def train_isolation_forest(X_train):
  param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 256, 512],
    'contamination': ['auto', 0.01, 0.05],
    'max_features': [1.0, 0.5, 0.25]
  }
  grid = GridSearchCV(IsolationForest(), param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train)

  isoF_model = grid.best_estimator_
  save_model(isoF_model, "isolation_forest")

  return isoF_model

# NEURAL NETWORK
def train_neural_network(X_train, y_train):
  param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'learning_rate_init': [0.001, 0.01]
  }
  grid = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train, y_train)

  neuralN_model = grid.best_estimator_
  save_model(neuralN_model, "neural_network")

  return neuralN_model

# COMBINING MODELS USING ENSEMBLE LEARNING
def train_ensemble(X_train, y_train):
  models = [
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('mlp', MLPClassifier(max_iter=1000))
  ]
  ensemble_model = VotingClassifier(estimators=models, voting='soft')

  param_grid = {
    'lr__C': [1, 10],
    'rf__n_estimators': [100, 200],
    'gb__n_estimators': [50, 100],
    'svm__C': [0.1, 1],
    'svm__Kernel': ['linear', 'rbf'],
    'mlp__hidden_layer_sizes': [(50,), (100,)]
  }

  grid = GridSearchCV(ensemble_model, param_grid, refit=True, verbose=2, n_jobs=-1)
  grid.fit(X_train, y_train)

  ens_model = grid.best_estimator_
  save_model(ens_model, "ensemble_learning")

  return ens_model