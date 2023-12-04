import os
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the Mushroom model
model_mushroom_path = 'mushroom_classifier_model.pkl'
with open(model_mushroom_path, 'rb') as model_file_mushroom:
    model_mushroom = pickle.load(model_file_mushroom)

# Load the Mushroom dataset
filename_mushroom = 'mushrooms.csv'
dataframe_mushroom = pd.read_csv(filename_mushroom)

# Separate features (X_mushroom) and target variable (y_mushroom)
X_mushroom = dataframe_mushroom.drop('class', axis=1)
y_mushroom = dataframe_mushroom['class']

# Convert categorical variables into numerical format using one-hot encoding
X_encoded_mushroom = pd.get_dummies(X_mushroom)

# Load the Valorant model
model_valo_path = 'valorant_model.pkl'
with open(model_valo_path, 'rb') as model_file_valo:
    model_valo = pickle.load(model_file_valo)

# Load the Valorant dataset
filename_valo = 'valorant.csv'
dataframe_valo = pd.read_csv(filename_valo)

# Separate features (X_valo) and target variable (y_valo)
X_valo = dataframe_valo.drop('Elo-Gain', axis=1)

# Function to make predictions using the Mushroom model
def predict_mushroom_class(features):
    # Convert the input features into a DataFrame
    input_data = pd.DataFrame([features], columns=X_mushroom.columns)

    # One-hot encode the input data with the columns used during training
    input_data_encoded = pd.get_dummies(input_data).reindex(columns=X_encoded_mushroom.columns, fill_value=0)

    # Use the trained model to make predictions
    prediction = model_mushroom.predict(input_data_encoded)

    return prediction[0]

# Function to make predictions using the Valorant model
def predict_elo_gain(features):
    # Convert the input features into a DataFrame
    input_data = pd.DataFrame([features], columns=X_valo.columns)

    # Use the trained model to make predictions
    prediction = model_valo.predict(input_data)

    return int(prediction[0])  # Convert prediction to an integer

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_mushroom():
    # Get input features from the form
    features = [request.form[field] for field in X_mushroom.columns]

    # Use the model to make predictions
    predicted_class = predict_mushroom_class(features)

    # Map predicted class to human-readable labels
    class_labels = {'e': 'edible', 'p': 'poisonous'}
    predicted_label = class_labels.get(predicted_class, 'unknown')

    return render_template('index.html', prediction=predicted_class)

@app.route('/predict_valo', methods=['POST'])
def predict_valo():
    # Get input features from the form
    features = [float(request.form[field]) for field in X_valo.columns]

    # Use the model to make predictions
    predicted_elo_gain = predict_elo_gain(features)

    return render_template('valorant.html', prediction2=predicted_elo_gain)

@app.route('/valorant')
def valo():
    return render_template('valorant.html')

if __name__ == '__main__':
    app.run(debug=True)
