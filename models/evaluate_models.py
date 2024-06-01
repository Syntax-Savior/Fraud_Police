
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# DEFINING A FUNCTION FOR MODEL EVALUATION.
def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)

  y_test = y_test.apply(lambda x: 1 if x > 0 else 0)
  y_pred = [1 if x > 0 else 0 for x in y_pred]

  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='binary')
  recall = recall_score(y_test, y_pred, average='binary')
  f1 = f1_score(y_test, y_pred, average='binary')
  return accuracy, precision, recall, f1

# DEFINING A FUNCTION TO PRINT EVALUATION RESULT.
def print_evaluation_results(model_name, accuracy, precision, recall, f1):
  print(f"{model_name} Performance:")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1 Score: {f1:.4f}\n")