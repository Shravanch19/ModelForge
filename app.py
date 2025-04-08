from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from function.generator import Generator, Column_names

app = Flask(__name__,static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'


# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html', static_url_path='/static')



@app.route('/generate', methods=['POST'])
def generate():
    # Ensure all form data is present
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    if 'target' not in request.form or 'algorithm' not in request.form:
        return jsonify({"error": "Target or algorithm missing"}), 400

    uploaded_file = request.files['file']
    target = request.form['target']
    algorithm = request.form['algorithm']
    print(uploaded_file)
    print(target)
    print(algorithm)

    try:
        # Call model training function
        result = Generator(uploaded_file, target, algorithm)
        # Convert Model objects to dictionary format for JSON serialization
        serialized_results = [{
            'model_name': model.model.__class__.__name__,
            'metric_value': model.metric_value,
            'code_snippets': model.code_snippets,
            'image': model.image
        } for model in result]
        return jsonify(serialized_results)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/retrieve_column_names', methods=['POST'])
def retrieve_column_names():
    print("retrieving column names")

    print(request.files)
    # Ensure a file is provided
    if 'file' not in request.files:
        print("No file provided")
        return jsonify({'error': 'No file provided'}), 400

    uploaded_file = request.files['file']

    try:
        print(uploaded_file)
        column_names = Column_names(uploaded_file)  # Assuming this processes the file
        return jsonify({'column_names': column_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True) 