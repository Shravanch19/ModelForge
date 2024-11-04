import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE

def Code_Generator(file, target_column):
    global code
    code = []
    code.append(base_code())
    print("base code generated")
    data = pd.read_csv(file)
    print("data imported")

    print("starting preprocessing")
    data = preprocessing(data, target_column)

    print("spliting the data")
    x_train, x_test, y_train, y_test = data_splitting_function(data, target_column)

    print("creating the model")
    model = model_training_function(x_train, y_train)

    print("checking the accuracy")
    acc = accuracy_checker_function(model, x_test, y_test)
    
    return data, "\n".join(code), acc

def base_code():
    return """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('path_to_file')
"""

def preprocessing(data, target):

    print("checking for missing values")
    null_function(data)

    print("checking for duplicates")
    duplicates_function(data)

    print("handling categorical data")
    handling_categorical_data_function(data)

    print("handeling imbalance data")
    data = imbalance_data_function(data, target)

    print("normalization of data")
    normalization_of_data_function(data)
    return data

def null_function(data):
    code.append("data.isnull().sum()")

    if data.isnull().values.any():
        missing_percentage = data.isnull().mean() * 100
        for column in data.columns:
            if missing_percentage[column] > 0:
                if missing_percentage[column] <= 7:
                    code.append(f"data.dropna(subset=['{column}'], inplace=True)")
                    data.dropna(subset=[column], inplace=True)
                else:
                    if data[column].dtype == 'object':
                        code.append(f"data['{column}'].fillna(data['{column}'].mode()[0], inplace=True)")
                        data[column].fillna(data[column].mode()[0], inplace=True)
                    else:
                        skewness = data[column].skew()
                        if abs(skewness) < 0.6:
                            code.append(f"data['{column}'].fillna(data['{column}'].mean(), inplace=True)")
                            data[column].fillna(data[column].mean(), inplace=True)
                        else:
                            code.append(f"data['{column}'].fillna(data['{column}'].median(), inplace=True)")
                            data[column].fillna(data[column].median(), inplace=True)

def duplicates_function(data):
    if data.duplicated().any():
        code.append("data = data.drop_duplicates()")
        data = data.drop_duplicates()


def handling_categorical_data_function(data):

    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtypes == 'object':
            data[col] = le.fit_transform(data[col])
    code.append(f"from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nfor col in data.columns:\n    if data[col].dtypes == 'object':\n        data[col] = le.fit_transform(data[col])")

def imbalance_data_function(data, target):
    if target in data.columns:
        smote = SMOTE()
        X = data.drop(target, axis=1)
        y = data[target]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        data = pd.concat([X_resampled, y_resampled], axis=1)
        code.append(f"from imblearn.over_sampling import SMOTE\nsmote = SMOTE()\nX_resampled, y_resampled = smote.fit_resample(X, y)")
    return data

def normalization_of_data_function(data, threshold=10):
    feature_ranges = np.ptp(data, axis=0)
    if np.max(feature_ranges) / np.min(feature_ranges) > threshold:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        code.append("from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndata = scaler.fit_transform(data)")

def data_splitting_function(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    code.append("x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)")
    return x_train, x_test, y_train, y_test

def model_training_function(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    code.append("from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(x_train, y_train)")
    return model

def accuracy_checker_function(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = r2_score(y_test, y_pred)
    code.append("from sklearn.metrics import r2_score\ny_pred = model.predict(x_test)\naccuracy = r2_score(y_test, y_pred)")
    return accuracy

data, Code, acc = Code_Generator('Salary_Data.csv', "Salary")
print("accuracy : ", end="")
print(acc)
print()
print(Code)