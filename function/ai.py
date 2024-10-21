import pandas as pd
import numpy as np

def Code_Genrator(file):

    code = base_code()
    data = pd.read_csv(file)

    preprocessing(data)
    pass

def base_code():
    return """
    import pandas as pd
    import numpy as np

    data = pd.read_csv(path_to_file)

    """

def preprocessing(data):
    
    null_function()
    duplicates_function()
    handeling_catogorical_data_function()
    imbalanced_data_function()
    normailization_of_data_function()
    feature_removing_function()
    return data

def null_function():
    # Check if there are missing values
    if data.isnull().values.any():
        # Get percentage of missing values for each column
        missing_percentage = data.isnull().mean() * 100
        
        for column in data.columns:
            if missing_percentage[column] > 0:
                if 5 <= missing_percentage[column] <= 10:
                    # Drop rows with missing values if missing data is between 5-10%
                    data.dropna(subset=[column], inplace=True)
                    code.append(f"data.dropna(subset=['{column}'], inplace=True)\n")
                
                else:
                    if data[column].dtype == 'object':
                        # Impute categorical data with mode
                        mode_value = data[column].mode()[0]
                        data[column].fillna(mode_value, inplace=True)
                        code.append(f"data['{column}'].fillna(data['{column}'].mode()[0], inplace=True)\n")
                    
                    else:
                        # Check if the data distribution is normal
                        skewness = data[column].skew()
                        if abs(skewness) < 0.5:
                            # If distribution is normal, impute with mean
                            mean_value = data[column].mean()
                            data[column].fillna(mean_value, inplace=True)
                            code.append(f"data['{column}'].fillna(data['{column}'].mean(), inplace=True)\n")
                        else:
                            # Otherwise, impute with median
                            median_value = data[column].median()
                            data[column].fillna(median_value, inplace=True)
                            code.append(f"data['{column}'].fillna(data['{column}'].median(), inplace=True)\n")

    # Check final missing values
    code.append("data.isnull().sum()\n")

def duplicates_function():
    if data.duplicated().any():
        data = data.drop_duplicates()
        code.append("data = data.drop_duplicates()\n")
    

def handeling_catogorical_data_function():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for i in data.columns:
        if data[i].dtypes == 'object':
            data[i] = le.fit_transform(data[i])
    
    code.append("""from sklearn.preprocessing import LabelEncoder\n
    le = LabelEncoder()\n
    for i in data.columns:\n
        if data[i].dtypes == 'object':\n
            data[i] = le.fit_transform(data[i])\n
    """)

def imbalanced_data_function():
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()

def normailization_of_data_function():
    pass
def feature_removing_function():
    pass

def data_spliting_function(target):
    from sklearn.model_selection import train_test_split
    X = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def model_tranning_function():
    pass
def accuracy_checker_function():
    pass