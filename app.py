# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:45:13 2020

@author: Suhas
"""

# Importing essential libraries

from flask import Flask,render_template,url_for,request
import pandas as pd 
import os
import pickle

import numpy as np

# Load the Random Forest CLassifier model
model = 'xgboost_random_model.pkl'
regressor = pickle.load(open(model, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        avg_temp = float(request.form['Average_temp'])
        max_temp = float(request.form['Max_temp'])
        min_temp = float(request.form['Min_temp'])
        at_pres = float(request.form['Atmospheric_pressure'])
        avg_hum = float(request.form['Average_humidity'])
        avg_vis = float(request.form['Average_visibility'])
        avg_speed = float(request.form['Average_windspeed'])
        max_sustained = float(request.form['Max sustained wind speed'])
        
        data = np.array([[avg_temp,max_temp, min_temp, at_pres, avg_hum, avg_vis, avg_speed, max_sustained]])
        my_prediction = regressor.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)
