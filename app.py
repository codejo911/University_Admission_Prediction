
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.json

        gre_score = float(data['gre_score'])
        toefl_score = float(data['toefl_score'])
        university_rating = float(data['university_rating'])
        sop = float(data['sop'])
        lor = float(data['lor'])
        cgpa = float(data['cgpa'])
        research = 1 if data['research'] == "yes" else 0

        # Load scaler
        scaler = pickle.load(open("scaling_model.pkl", "rb"))
        scaled_input = scaler.transform([[gre_score, toefl_score, university_rating,
                                          sop, lor, cgpa, research]])

        # Load model
        model = pickle.load(open("ridge_regression_model.pkl", "rb"))
        prediction = model.predict(scaled_input)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app
