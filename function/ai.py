import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

class ModelInfo:
    def __init__(self, name, accuracy, code, score_plot):
        self.name = name
        self.accuracy = accuracy
        self.code = code
        self.score_plot = score_plot

    def to_dict(self):
        return {
            'model_name': self.name,
            'accuracy': self.accuracy,
            'code': self.code,
            'score_plot': self.score_plot
        }

def colums_printer(file):
    data = pd.read_csv(file)
    return data.columns

def code_generator(file, target_column):
    code_snippets = []
    data = pd.read_csv(file)
    code_snippets.append(base_code())
    data = preprocessing(data, target_column, code_snippets)
    plot1 = heatMap(data)
    x_train, x_test, y_train, y_test = data_splitting_function(data, target_column, code_snippets)
    model_infos = model_training_and_evaluation(x_train, y_train, x_test, y_test, code_snippets)
    return [model.to_dict() for model in model_infos], plot1

def base_code():
    return """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('path_to_file')
"""

def preprocessing(data, target, code_snippets):
    null_function(data, code_snippets)
    duplicates_function(data, code_snippets)
    handling_categorical_data_function(data, code_snippets)
    normalization_of_data_function(data, code_snippets)
    return data

def heatMap(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return base64_image


def null_function(data, code_snippets):
    code_snippets.append("data.isnull().sum()")
    if data.isnull().values.any():
        missing_percentage = data.isnull().mean() * 100
        for column in data.columns:
            if missing_percentage[column] > 0:
                if missing_percentage[column] <= 7:
                    code_snippets.append(f"data.dropna(subset=['{column}'], inplace=True)")
                    data.dropna(subset=[column], inplace=True)
                else:
                    if data[column].dtype == 'object':
                        code_snippets.append(f"data['{column}'].fillna(data['{column}'].mode()[0], inplace=True)")
                        data[column].fillna(data[column].mode()[0], inplace=True)
                    else:
                        skewness = data[column].skew()
                        if abs(skewness) < 0.6:
                            code_snippets.append(f"data['{column}'].fillna(data['{column}'].mean(), inplace=True)")
                            data[column].fillna(data[column].mean(), inplace=True)
                        else:
                            code_snippets.append(f"data['{column}'].fillna(data['{column}'].median(), inplace=True)")
                            data[column].fillna(data[column].median(), inplace=True)

def duplicates_function(data, code_snippets):
    if data.duplicated().any():
        code_snippets.append("data = data.drop_duplicates()")
        data = data.drop_duplicates()

def handling_categorical_data_function(data, code_snippets):
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtypes == 'object':
            data[col] = le.fit_transform(data[col])
    code_snippets.append("from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nfor col in data.columns:\n    if data[col].dtypes == 'object':\n        data[col] = le.fit_transform(data[col])")

def normalization_of_data_function(data, code_snippets, threshold=10):
    feature_ranges = np.ptp(data, axis=0)
    if np.max(feature_ranges) / np.min(feature_ranges) > threshold:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        code_snippets.append("from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndata = scaler.fit_transform(data)")

def data_splitting_function(data, target, code_snippets):
    X = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    code_snippets.append("x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)")
    return x_train, x_test, y_train, y_test

def model_training_and_evaluation(x_train, y_train, x_test, y_test, code_snippets):
    models = {
        'Linear Regression': (LinearRegression(), "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()"),
        'Ridge Regression': (Ridge(), "from sklearn.linear_model import Ridge\nmodel = Ridge()"),
        'Lasso Regression': (Lasso(), "from sklearn.linear_model import Lasso\nmodel = Lasso()"),
        'Decision Tree Regressor': (DecisionTreeRegressor(), "from sklearn.tree import DecisionTreeRegressor\nmodel = DecisionTreeRegressor()"),
        'Random Forest Regressor': (RandomForestRegressor(), "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()")
    }
    
    model_infos = []
    for model_name, (model, model_code) in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = r2_score(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"R2 Score Scatter Plot for {model_name} (R2 Score: {accuracy:.2f}%)")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()


        model_infos.append(ModelInfo(model_name, accuracy,( "\n".join(code_snippets + [model_code] + [f"model.fit(x_train, y_train)\ny_pred = model.predict(x_test)\naccuracy = r2_score(y_test, y_pred)"])), base64_image))
    
    return model_infos

# def Predict(file,target):
#     data = pd.read_csv(file)
#     code_snippets = []
#     data = preprocessing(data, target, code_snippets)

#     x_train, x_test, y_train, y_test = data_splitting_function(data, target, code_snippets)
#     model = RandomForestRegressor()
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     accuracy = r2_score(y_test, y_pred)
#     print(accuracy)
#     user_input = (28,"Male","PhD","Director","10")
#     pridected = model.predict(user_input)
#     print(pridected)

# Predict("Salary_Data.csv","Salary")