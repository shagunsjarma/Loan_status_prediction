import os
import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, input_data):
        try:
            # Accept input as dict or DataFrame
            if isinstance(input_data, dict):
                data = pd.DataFrame([input_data])
            elif isinstance(input_data, pd.DataFrame):
                data = input_data.copy()
            else:
                raise ValueError('Input data must be a dict or pandas DataFrame')

            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Transform input data
            data_transformed = preprocessor.transform(data)
            # Predict
            prediction = model.predict(data_transformed)
            return prediction
        except Exception as e:
            raise CustomException(str(e), sys)

class CustomData:
    def __init__(self, 
                 gender:str, 
                 married:str, 
                 dependents:str, 
                 education:str, 
                 self_employed:str, 
                 applicantincome:int, 
                 coapplicantincome:int, 
                 loanamount:int, 
                 loan_amount_term:int, 
                 credit_history:int, 
                 property_area:str):
        self.gender = gender
        self.married = married
        self.dependents = dependents
        self.education = education
        self.self_employed = self_employed
        self.applicantincome = applicantincome
        self.coapplicantincome = coapplicantincome
        self.loanamount = loanamount
        self.loan_amount_term = loan_amount_term
        self.credit_history = credit_history
        self.property_area = property_area
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender': [self.gender],
                'Married': [self.married],
                'Dependents': [self.dependents],
                'Education': [self.education],
                'Self_Employed': [self.self_employed],
                'ApplicantIncome': [self.applicantincome],
                'CoapplicantIncome': [self.coapplicantincome],
                'LoanAmount': [self.loanamount],
                'Loan_Amount_Term': [self.loan_amount_term],
                'Credit_History': [self.credit_history],
                'Property_Area': [self.property_area]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(str(e), sys)
