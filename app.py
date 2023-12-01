import os
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Set the path to the directory containing the model
model_dir = os.path.join(os.getcwd(), 'models')

# Load the model with the correct path
model_path = 'mushroom_classifier_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset
filename = 'mushrooms.csv'
dataframe = pd.read_csv(filename)

# Separate features (X) and target variable (y)
X = dataframe.drop('class', axis=1)
y = dataframe['class']

# Convert categorical variables into numerical format using one-hot encoding
X_encoded = pd.get_dummies(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/try')
def try_page():
    return render_template('try.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = [request.form[field] for field in X.columns]

    # Use the model to make predictions
    predicted_class = predict_mushroom_class(features)

    # Map predicted class to human-readable labels
    class_labels = {'e': 'edible', 'p': 'poisonous'}
    predicted_label = class_labels.get(predicted_class, 'unknown')

    return render_template('index.html', prediction=predicted_class)

def predict_mushroom_class(features):
    # Convert the input features into a DataFrame
    input_data = pd.DataFrame([features], columns=X.columns)

    # One-hot encode the input data with the columns used during training
    input_data_encoded = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)

    # Use the trained model to make predictions
    prediction = model.predict(input_data_encoded)

    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
