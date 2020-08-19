from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
from joblib import load
import os

app = Flask(__name__)  

@app.route("/" , methods=['GET', 'POST'])
def index(): 
        result = "result appears here"
        if request.method == 'POST':
                if "datainput" in request.form:
                        data = []
                        pregnancies = request.form['pregnancies']
                        data.append(float(pregnancies))
                        glucose = request.form['glucose']
                        data.append(float(glucose))
                        bloodpressure = request.form['bloodpressure']
                        data.append(float(bloodpressure))
                        skinthickness = request.form['skinthickness']
                        data.append(float(skinthickness))
                        insulin = request.form['insulin']
                        data.append(float(insulin))
                        bmi = request.form['bmi']
                        data.append(float(bmi))
                        diabetespedigreefunction = request.form['diabetespedigreefunction']
                        data.append(float(diabetespedigreefunction))
                        age = request.form['age']
                        data.append(float(age))

                        file_path = os.path.join(os.path.dirname(__file__), 'diabetesdetectionmodel.pkl')
                        diabetes_model = load(file_path)

                        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                        dictionary = dict()
                        for value,col in zip(data,columns):
                            dictionary[col] = [value]
                        print(data)
                        dataframe = pd.DataFrame(dictionary)

                        prediction = diabetes_model.predict(dataframe)
                        print('__________________________________')
                        print(prediction)
                        print('__________________________________')

                        result = ('positive' if prediction[0] == 1 else 'negative')
                        
                        
        # preprocessing here
        
        return render_template("mainpage.html", result = result)
