# Loan Status Prediction

A complete machine learning pipeline and web application for predicting loan approval status based on applicant data. This project demonstrates a modular, production-ready approach to data ingestion, transformation, model training, and deployment with a user-friendly web interface.

---

## Features
- **End-to-end ML pipeline**: Data ingestion, preprocessing, model selection, and prediction.
- **Multiple models**: Trains and selects the best from Logistic Regression, Decision Tree, Random Forest, AdaBoost, and XGBoost.
- **Reusable components**: Modular code for easy maintenance and extension.
- **Web interface**: Simple Flask app for user input and prediction display.
- **Artifact management**: Saves preprocessor and model for reproducible predictions.

---

## Project Structure
```
loanstatus/
│
├── data/                  # Raw data (loan_data.csv)
├── models/                # (empty, for future use)
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── components/        # Data ingestion, transformation, model training
│   ├── pipeline/          # Predict and train pipeline scripts
│   ├── utils.py           # Utility functions (save/load objects, etc.)
│   ├── logger.py, exception.py
│
├── templates/
│   └── index.html         # Main web form
│
├── app.py                 # Flask app
├── requirements.txt
└── setup.py
```

---

## Setup Instructions

### 1. **Clone the Repository**
```
git clone <repo-url>
cd loanstatus
```

### 2. **Create and Activate a Virtual Environment**
```
python -m venv loanenv
loanenv\Scripts\activate   # On Windows
# or
source loanenv/bin/activate # On Mac/Linux
```

### 3. **Install Dependencies**
```
pip install -r requirements.txt
```

### 4. **Prepare Data**
- Ensure `data/loan_data.csv` exists (already included in the repo).

---

## Usage

### **A. Train the Model**
From the project root, run:
```
python -m src.pipeline.train_pipeline
```
- This will ingest data, preprocess, train models, and save the best model and preprocessor in the `artifacts/` directory.

### **B. Run the Web Application**
```
python app.py
```
- Open your browser and go to `http://localhost:5000` (or the address shown in your terminal).
- Fill out the form and submit to get a loan approval prediction.

---

## How It Works
1. **Data Ingestion**: Reads and splits the raw data.
2. **Data Transformation**: Preprocesses features (imputation, scaling, encoding).
3. **Model Training**: Trains several models, selects and saves the best.
4. **Prediction Pipeline**: Loads artifacts, transforms user input, and predicts.
5. **Web App**: Collects user input, calls the prediction pipeline, and displays results.

---

## Extending the Project
- Add new models or features in `src/components/model_trainer.py`.
- Improve preprocessing in `src/components/data_transformation.py`.
- Enhance the web UI in `templates/index.html`.
- Deploy with Docker, Gunicorn, or to cloud platforms.

---

## License
This project is for educational and demonstration purposes.
