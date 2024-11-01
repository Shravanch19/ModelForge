import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def Code_Generator(file, target_column):
    code = base_code()
    data = pd.read_csv(file)

    data = preprocessing(data)
    x_train, x_test, y_train, y_test = data_splitting_function(data, target_column)
    model = model_training_function(x_train, y_train)
    accuracy_checker_function(model, x_test, y_test)
    
    return data, code

def base_code():
    return """
    import pandas as pd
    import numpy as np

    data = pd.read_csv(path_to_file)

    """

def preprocessing(data):
    data = null_function(data)
    data = duplicates_function(data)
    data = handling_categorical_data_function(data)
    data = imbalance_data_function(data)
    data = normalization_of_data_function(data)
    data = feature_removal_function(data)
    return data

def null_function(data):
    # Check if there are missing values
    if data.isnull().values.any():
        # Get percentage of missing values for each column
        missing_percentage = data.isnull().mean() * 100
        
        for column in data.columns:
            if missing_percentage[column] > 0:
                if 5 <= missing_percentage[column] <= 10:
                    # Drop rows with missing values if missing data is between 5-10%
                    data.dropna(subset=[column], inplace=True)
                else:
                    if data[column].dtype == 'object':
                        # Impute categorical data with mode
                        data[column].fillna(data[column].mode()[0], inplace=True)
                    else:
                        # Impute numerical data based on distribution skewness
                        skewness = data[column].skew()
                        if abs(skewness) < 0.5:
                            data[column].fillna(data[column].mean(), inplace=True)
                        else:
                            data[column].fillna(data[column].median(), inplace=True)
    return data

def duplicates_function(data):
    if data.duplicated().any():
        data = data.drop_duplicates()
    return data

def handling_categorical_data_function(data):
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtypes == 'object':
            data[col] = le.fit_transform(data[col])
    return data

def imbalance_data_function(data):
    # Handling imbalanced data if the target variable is specified
    target_column = 'target'  # Replace with the actual target column
    if target_column in data.columns:
        smote = SMOTE()
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        data = pd.concat([X_resampled, y_resampled], axis=1)
    return data

def normalization_of_data_function(data, method="standard"):
    # Normalizing data
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def feature_removal_function(data):
    # Remove features with low variance
    low_variance_cols = [col for col in data.columns if data[col].nunique() < 2]
    data = data.drop(columns=low_variance_cols)
    return data

def data_splitting_function(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def model_training_function(x_train, y_train):
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train, y_train)
    return model

def accuracy_checker_function(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Additional function: feature selection using correlation
def feature_selection_function(data, threshold=0.8):
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    data = data.drop(columns=to_drop)
    return data
