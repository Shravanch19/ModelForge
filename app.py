from flask import Flask, render_template, jsonify

app = Flask(__name__,static_url_path='/assets')

@app.route("/")
def hello_world():
    return render_template('main.html')

@app.route('/get_message', methods=['GET'])
def get_message():
    message = "Code Genrated for the dataset!!"
    return jsonify({'message': message})