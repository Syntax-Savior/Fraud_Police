from utils.data_loader import (
    load_credit_data, load_creditcard_2023, load_card_fraud, load_credit_fraud, 
    frequency_encode, scale_data, apply_smote, split_data
)

from models.train_models import (
    train_logistic_regression, train_random_forest, train_gradient_boosting, 
    train_svm, train_isolation_forest, train_neural_network, train_ensemble
)

from models.evaluate_models import evaluate_model, print_evaluation_results

def ensure_binary_labels(y):
    y = y.apply(lambda x: 1 if x > 0 else 0)
    return y

def main():
    df1 = load_credit_data()
    df2 = load_creditcard_2023()
    df3 = load_card_fraud()
    df4 = load_credit_fraud()

    # PRINTING THE SHAPE OF EACH DATASET TO CONFIRM SUCCESSFUL LOADING.
    #print("credit_data.csv:", df1.shape)
    #print("creditcard_2023.csv:", df2.shape)
    #print("card_fraud.csv:", df3.shape)
    #print("credit_fraud.csv:", df4.shape)

    # CHECK MISSING VALUES IN ALL DATASETS
    #check_missing_values(df1, "credit_data.csv")
    #check_missing_values(df2, "creditcard_2023.csv")
    #check_missing_values(df3, "card_fraud.csv")
    #check_missing_values(df4, "credit_fraud.csv")

    # ENCODE CATEGORICAL VARIABLES
    df3_encoded = frequency_encode(df3)
    df4_encoded = frequency_encode(df4)

    # PRINTING THE SHAEP OF THE ENCODED CATEGORICAL VARIABLES TO CONFIRM SUCCESSFUL ENCODING.
    #print("card_fraud.csv encoded shape:", df3_encoded.shape)
    #print("credit_fraud.csv encoded shape:", df4_encoded.shape)

    df1_scaled = scale_data(df1)
    df2_scaled = scale_data(df2)
    df3_scaled = scale_data(df3_encoded)
    df4_scaled = scale_data(df4_encoded)

    # PRINTING THE SHAPE OF EACH SCALED DATASET TO CONFIRM SUCCESSFUL SCALING.
    #print("credit_data.csv scaled shape:", df1_scaled.shape)
    #print("creditcard_2023.csv scaled shape:", df2_scaled.shape)
    #print("card_fraud.csv scaled shape:", df3_scaled.shape)
    #print("credit_fraud.csv scaled shape:", df4_scaled.shape)

    X_train1, X_test1, y_train1, y_test1 = split_data(df1_scaled, target_column='Class')
    X_train2, X_test2, y_train2, y_test2 = split_data(df2_scaled, target_column='Class')
    X_train3, X_test3, y_train3, y_test3 = split_data(df3_scaled, target_column='is_fraud')
    X_train4, X_test4, y_train4, y_test4 = split_data(df4_scaled, target_column='is_fraud')

    y_train1 = ensure_binary_labels(y_train1)
    y_test1 = ensure_binary_labels(y_test1)
    y_train2 = ensure_binary_labels(y_train2)
    y_test2 = ensure_binary_labels(y_test2)
    y_train3 = ensure_binary_labels(y_train3)
    y_test3 = ensure_binary_labels(y_test3)
    y_train4 = ensure_binary_labels(y_train4)
    y_test4 = ensure_binary_labels(y_test4)

    # APPLIED SMOTE TO HANDLE IMBALANCED DATA
    X_train1_res, y_train1_res = apply_smote(X_train1, y_train1)
    
    logistic_regression_model = train_logistic_regression(X_train1_res, y_train1_res)
    random_forest_model = train_random_forest(X_train1_res, y_train1_res)
    gradient_boosting_model = train_gradient_boosting(X_train1_res, y_train1_res)
    svm_model = train_svm(X_train1_res, y_train1_res)
    isolation_forest_model = train_isolation_forest(X_train1_res)
    neural_network_model = train_neural_network(X_train1_res, y_train1_res)
    ensemble_model = train_ensemble(X_train1_res, y_train1_res)

    models = {
        "Logistic Regression": logistic_regression_model,
        "Random Forest": random_forest_model,
        "Gradient Boosting": gradient_boosting_model,
        "SVM": svm_model,
        "Isolation Forest": isolation_forest_model,
        "Neural Network": neural_network_model,
        "Ensemble": ensemble_model
    }

    for model_name, train_func in models.items():
        if model_name == "Isolation Forest":
            model = train_func(X_train1_res)
            accuracy, precision, recall, f1 = evaluate_model(model, X_test1, y_test1)
        else:
            model = train_func(X_train1_res, y_train1_res)
            accuracy, precision, recall, f1 = evaluate_model(model, X_test1, y_test1)
        print_evaluation_results(model_name, accuracy, precision, recall, f1)

if __name__ == "__main__":
    main()