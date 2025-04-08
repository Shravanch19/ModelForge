import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from scipy import stats


def Column_names(data):
    data = pd.read_csv(data)
    return data.columns.tolist()

class Model:
    def __init__(self, model, metric_value, code_snippets, image):
        self.model = model
        self.metric_value = metric_value
        self.code_snippets = code_snippets
        self.image = image

def Generator(data, target, algorithm):

    code = []
    code.append(base_code())
    data = pd.read_csv(data)
    data = preprocess(data, code, target, algorithm)
    x_train, x_test, y_train, y_test = split_function(data, target)
    print(x_train.head())
    print("________________")
    print(y_test)
    print("________________")
    model_info = model_function(algorithm, x_train, x_test, y_train, y_test, code)
    return model_info

def base_code():
    return """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    """

def preprocess(data, code, target, algorithm):
    print('checking for duplicates')
    data = duplicate_function(data, code)
    print('checking for null values')
    data = null_function(data, code)
    # print('checking for outliers')
    # data = handle_outliers(data, code)
    print('encoding')
    data = encode_function(data, code)
    print('scaling')
    data = scale_function(data, code, target, algorithm)
    return data

def null_function(data, code_snippets, drop_threshold=7, skewness_threshold=0.6):
    # Track missing values
    code_snippets.append("data.isnull().sum()")

    if not data.isnull().values.any():
        return data  # Exit early if no missing values

    missing_percentage = data.isnull().mean() * 100

    for column in data.columns:
        missing_pct = missing_percentage[column]

        if missing_pct == 0:
            continue

        # Drop rows if missing percentage is below threshold
        if missing_pct <= drop_threshold:
            code_snippets.append(f"data.dropna(subset=['{column}'], inplace=True)")
            data.dropna(subset=[column], inplace=True)
            continue

        # Fill missing values based on data type and skewness
        dtype = data[column].dtype

        if dtype == 'object':
            mode_value = data[column].mode()[0]
            code_snippets.append(f"data['{column}'].fillna('{mode_value}', inplace=True)")
            data[column].fillna(mode_value, inplace=True)
        else:
            skewness = data[column].skew()
            if abs(skewness) < skewness_threshold:
                mean_value = data[column].mean()
                code_snippets.append(f"data['{column}'].fillna({mean_value}, inplace=True)")
                data[column].fillna(mean_value, inplace=True)
            else:
                median_value = data[column].median()
                code_snippets.append(f"data['{column}'].fillna({median_value}, inplace=True)")
                data[column].fillna(median_value, inplace=True)

def duplicate_function(data, code_snippets):
    if not data.duplicated().values.any():
        return data
    data.drop_duplicates(inplace=True)
    code_snippets.append("data.duplicated().sum()")
    code_snippets.append("data.drop_duplicates(inplace=True)")
    print('dropping duplicates...........')
    return data


# def handle_outliers(csv_path, method='remove', z_threshold=3):
#     """
#     Handles outliers in a CSV dataset using the Z-score method.
    
#     Parameters:
#         csv_path (str): Path to the CSV file.
#         method (str): 'remove' to drop outliers, 'replace' to replace them with the median.
#         z_threshold (float): Threshold for Z-score (default is 3).
    
#     Returns:
#         pd.DataFrame: Processed DataFrame with outliers handled.
#     """
#     df = pd.read_csv(csv_path)
    
#     for col in df.select_dtypes(include=np.number).columns:
#         z_scores = np.abs(stats.zscore(df[col]))
        
#         if method == 'remove':
#             df = df[z_scores < z_threshold]
#         elif method == 'replace':
#             median_value = df[col].median()
#             df[col] = np.where(z_scores >= z_threshold, median_value, df[col])
    
#     return df


def encode_function(data, code_snippets):
    code_snippets.append("from sklearn.preprocessing import LabelEncoder")
    LE = LabelEncoder()
    code_snippets.append("LE = LabelEncoder()")
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = LE.fit_transform(data[column])
    
    code_snippets.append(f"for {column} in data.columns:")
    code_snippets.append(f"    if data['{column}'].dtype == 'object':")
    code_snippets.append(f"        data['{column}'] = LE.fit_transform(data['{column}'])")
    print('encoding done...........')
    return data


def scale_function(data, code_snippets, target, algorithm, threshold=10):
    code_snippets.append("from sklearn.preprocessing import StandardScaler")
    
    feature_ranges = np.ptp(data, axis=0)
    if np.min(feature_ranges) == 0 or np.max(feature_ranges) / np.min(feature_ranges) > threshold:
        scaler = StandardScaler()
        if algorithm == "Classification":
            for column in data.columns:
                if column != target:
                    data[column] = scaler.fit_transform(data[[column]])
                    code_snippets.append(f"data['{column}'] = scaler.fit_transform(data[['{column}']])")
        else:
            data[data.columns] = scaler.fit_transform(data)
            code_snippets.append("data[data.columns] = scaler.fit_transform(data)")

    return data



def split_function(data, target):
    x= data.drop(columns=[target], axis=1)
    y= data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print('splitting done...........')
    return x_train, x_test, y_train, y_test


def model_function(algorithm, x_train, x_test, y_train, y_test, code_snippets):
    # Define model mappings
    model_mapping = {
        "Regression": {
            "models": [
                (LinearRegression(), "from sklearn.linear_model import LinearRegression"),
                (RandomForestRegressor(), "from sklearn.ensemble import RandomForestRegressor"),
                (KNeighborsRegressor(), "from sklearn.neighbors import KNeighborsRegressor"),
                (DecisionTreeRegressor(), "from sklearn.tree import DecisionTreeRegressor")
            ],
            "metric": r2_score,
            "metric_name": "R2"
        },
        "Classification": {
            "models": [
                (LogisticRegression(), "from sklearn.linear_model import LogisticRegression"),
                (RandomForestClassifier(), "from sklearn.ensemble import RandomForestClassifier"),
                (KNeighborsClassifier(), "from sklearn.neighbors import KNeighborsClassifier"),
                (DecisionTreeClassifier(), "from sklearn.tree import DecisionTreeClassifier")
            ],
            "metric": accuracy_score,
            "metric_name": "Accuracy"
        }
    }

    if algorithm not in model_mapping:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    models_info = model_mapping[algorithm]

    models_results = []

    # Iterate through models, fit, predict, and evaluate
    for model, import_statement in models_info["models"]:
        print(model)
        print("________________")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Evaluate performance
        metric_value = models_info["metric"](y_test, y_pred)
        print(metric_value)
        print("________________")

        # Store code snippets
        model_code_snippets = [
            import_statement,
            f"model = {model.__class__.__name__}()",
            "model.fit(x_train, y_train)",
            "y_pred = model.predict(x_test)",
            f"from sklearn.metrics import {models_info['metric'].__name__}",
            f"{models_info['metric'].__name__} = {models_info['metric'].__name__}(y_test, y_pred)",
            f"print(f'{models_info['metric_name']}: {{{models_info['metric'].__name__}}}')"
        ]

        # Add model code snippets to the main code snippets list
        code_snippets.extend(model_code_snippets)

        if algorithm == "Regression":
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"R2 Score Scatter Plot for {model.__class__.__name__} (R2 Score: {metric_value:.2f}%)")
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            base64_image = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
        else:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual') 
            plt.title(f'Confusion Matrix for {model.__class__.__name__} (Accuracy: {metric_value:.2f}%)')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            base64_image = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

        #convert code_snippets to string
        codes = "\n".join(code_snippets)
        # Store model results
        models_results.append(Model(model, metric_value, codes, base64_image))

    return models_results

# Generator("D:\Codes\ModelForge2\Iris.csv", "Species", "Classification")
# print("done")
# Generator("D:\Codes\ModelForge2\Housing.csv", "price", "Regression")

