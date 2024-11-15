# def imbalance_data_function(data, target):
#     if target in data.columns:
#         smote = SMOTE()
#         X = data.drop(target, axis=1)
#         y = data[target]
#         X_resampled, y_resampled = smote.fit_resample(X, y)
#         data = pd.concat([X_resampled, y_resampled], axis=1)
#         code.append(f"from imblearn.over_sampling import SMOTE\nsmote = SMOTE()\nX_resampled, y_resampled = smote.fit_resample(X, y)")
#     return data

# print("handeling imbalance data")
# data = imbalance_data_function(data, target)

# accuracy
# : 
# 0.678901140584049
# code
# : 
# "\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\ndata = pd.read_csv('path_to_file')\n\ndata.isnull().sum()\ndata.dropna(subset=['Age'], inplace=True)\ndata.dropna(subset=['Gender'], inplace=True)\ndata.dropna(subset=['Education Level'], inplace=True)\ndata.dropna(subset=['Job Title'], inplace=True)\ndata.dropna(subset=['Years of Experience'], inplace=True)\ndata.dropna(subset=['Salary'], inplace=True)\ndata = data.drop_duplicates()\nfrom sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nfor col in data.columns:\n    if data[col].dtypes == 'object':\n        data[col] = le.fit_transform(data[col])\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndata = scaler.fit_transform(data)\nx_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\nfrom sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(x_train, y_train)\ny_pred = model.predict(x_test)\naccuracy = r2_score(y_test, y_pred)"
# model_name
# : 
# "Linear Regression"
# score_plot
