from flask import Flask, render_template, jsonify, request
from function.ai import colums_printer, code_generator

app = Flask(__name__,static_url_path='/assets')

@app.route("/")
def hello_world():
    return render_template('main.html')

@app.route('/get_message', methods=['POST'])
def get_message():
    file = request.files['file']
    target_variable = request.form['target_variable']
    code_snippets,heat = code_generator(file, target_variable)
    return jsonify({'code_snippets': code_snippets, 'heat': heat})

@app.route('/get_columns', methods=['POST'])
def get_columns():
    file = request.files['file']
    columns = colums_printer(file)
    columns_list = list(columns)
    return jsonify({'columns': columns_list})


