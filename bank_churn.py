import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


#initialize Flask
app = Flask(__name__)

#Loading the model file
model = joblib.load('final_model.pkl')

#to launch home page
@app.route("/churn")
def home():
    #Loading a HTML page
    return render_template('index.html')

#Prediction page
@app.route("/y_predict" ,methods=['POST'])
def y_predict():
    x_col=['CreditScore','Geography','Gender','Age','Tenure','Balance',
           'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
    
    #Get all the user entered values from the form
    data = [[x for x in request.form.values()]]
    
    print(data)
    
    data=pd.DataFrame(data,columns=x_col)
    
    #Prediction
    #Predict--Probabilities
    #predict_classes --0 or 1
    prediction = model.predict(data)
    #prediction 0--stayed  1--left the bank
    print(prediction)
    l=["Stayed","Exited"]
    #l[1], prediction[0][0]
    text="Prediction: "+l[prediction[0]]
    return render_template('index.html', prediction_text=text)
  
if __name__ == "__main__":

    app.run(debug=True)
    
    
    

