# Model Forge

Model Forge is a modern web-based tool for generating machine learning models from CSV data. It provides an intuitive interface for uploading datasets, selecting algorithms, and generating ready-to-use code and performance plots.

## Features

- **CSV Upload**: Drag and drop your CSV file to get started.
- **Algorithm Selection**: Choose between Regression and Classification tasks.
- **Feature Selection**: Select the target variable for your model.
- **Model Generation**: Automatically generates Python code for the best model based on your data and selected algorithm.
- **Performance Visualization**: View model performance plots directly in the browser.
- **Modern UI**: Responsive, glassmorphic design with smooth transitions and accessibility features.

## How It Works

1. **Upload CSV**: Click or drag your CSV file into the upload area.
2. **Select Algorithm**: Choose Regression or Classification.
3. **Select Target Variable**: Pick the column you want to predict.
4. **Generate Model**: Click "Generate Model" to receive code and performance plots.

## Developer

Developed by Shravan

---

**Note:** This is a frontend-only project. Backend endpoints (`/generate`, `/retrieve_column_names`) must be implemented separately for full model generation functionality.
