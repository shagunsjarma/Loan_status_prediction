from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            gender=request.form.get('gender') or "",
            married=request.form.get('married') or "",
            dependents=request.form.get('dependents') or "",
            education=request.form.get('education') or "",
            self_employed=request.form.get('self_employed') or "",
            applicantincome=int(request.form.get('applicantincome') or 0),
            coapplicantincome=int(request.form.get('coapplicantincome') or 0),
            loanamount=int(request.form.get('loanamount') or 0),
            loan_amount_term=int(request.form.get('loan_amount_term') or 0),
            credit_history=int(request.form.get('credit_history') or 0),
            property_area=request.form.get('property_area') or ""
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', final_result=results[0])



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)