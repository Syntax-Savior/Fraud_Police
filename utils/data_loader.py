import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# DEFINING THE PATH OF EACH DATASET.
file1_path = "D:/Academia/Programming_CS/0_RandomProjects/Fraud-Detection/venv/data/credit_data.csv"
file2_path = "D:/Academia/Programming_CS/0_RandomProjects/Fraud-Detection/venv/data/creditcard_2023.csv"
file3_path = "D:/Academia/Programming_CS/0_RandomProjects/Fraud-Detection/venv/data/card_fraud.csv"
file4_path = "D:/Academia/Programming_CS/0_RandomProjects/Fraud-Detection/venv/data/credit_fraud.csv"

# DEFINING FUNCTIONS TO READ EACH CSV FILE.
def load_credit_data():
    return pd.read_csv(file1_path, header=0)

def load_creditcard_2023():
    return pd.read_csv(file2_path, header=0)

def load_card_fraud():
    return pd.read_csv(file3_path, header=0, names=[
        "transaction_datetime", "cc_num", "merchant", "category", "amount", 
        "first_name", "last_name", "gender", "street", "city", "state", "zip", 
        "latitude", "longitude", "city_population", "job", "date_of_birth", 
        "transaction_number", "unix_timestamp", "merchant_latitude", 
        "merchant_longitude", "is_fraud"
    ])

def load_credit_fraud():
    return pd.read_csv(file4_path, header=0, names=[
        "transaction_datetime", "cc_num", "merchant", "category", "amount", 
        "first_name", "last_name", "gender", "street", "city", "state", "zip", 
        "latitude", "longitude", "city_population", "job", "date_of_birth", 
        "transaction_number", "unix_timestamp", "merchant_latitude", 
        "merchant_longitude", "is_fraud"
    ])

# DEFINING A FUNCTION TO CHECK MISSING VALUES.
def check_missing_values(df, name):
    missing_values = df.isnull().sum()
    print(f"Missing Values in {name}:\n{missing_values[missing_values > 0]}\n")

# DEFINING A FUNCTION TO ENCODE CATEGORICAL VARIABLES.
def frequency_encode(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        freq_encoding = df[column].value_counts() / len(df)
        df[column] = df[column].map(freq_encoding)
    return df

# DEFINING A FUNCTION TO SCALE THE DATA.
def scale_data(df):
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df

# DEFINING A FUNCTION TO HANDLE IMBALANCED DATA
def apply_smote(X_train, y_train):
    smote = SMOTE()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# DEFINING A FUNCTION TO SPLIT EACH DATASET INTO TRAINING AND TESTING SETS.
def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
