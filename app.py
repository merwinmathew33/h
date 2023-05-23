import math
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index.html')
def main_page():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self-employed'])
    ApplicantIncome = int(request.form['applicant-income'])
    CoapplicantIncome = int(request.form['co-applicant-income'])
    LoanAmount = int(request.form['loan-amount'])
    Loan_Amount_Term = int(request.form['loan-amount-term'])
    credit_history = int(request.form['credit-history'])
    property_area = int(request.form['property-area'])
    

    final_features = np.array([[gender, married, dependents, education, self_employed,ApplicantIncome,CoapplicantIncome, LoanAmount, Loan_Amount_Term, credit_history, property_area]])
    
    data = pd.DataFrame(final_features,index=[0])
    
    prediction = model.predict(data)
    output =  model.predict(data)
    print(output)
    if output == 1:
        return render_template('Congrats.html')
    else:
        return render_template('Denied.html')


if __name__ == "__main__":
    app.run(debug=True)