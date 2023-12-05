import os
import pandas as pd
from flask import Flask, render_template, request
import pickle
from werkzeug.utils import redirect
from flask_mysqldb import MySQL

#for chart
import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '123456'
app.config['MYSQL_DB'] = 'ml-algo'

mysql = MySQL(app)
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

@app.route('/eda')
def eda():
    
    # Fetch Mushroom data
    cur_mushroom = mysql.connection.cursor()
    cur_mushroom.execute("SELECT * FROM mushrooms")
    mushrooms_data = cur_mushroom.fetchall()
    cur_mushroom.close()

    # Fetch Valorant data
    cur_valorant = mysql.connection.cursor()
    cur_valorant.execute("SELECT * FROM valorant")
    valorant_data = cur_valorant.fetchall()
    cur_valorant.close()

    return render_template('eda.html', mushrooms=mushrooms_data, valorant=valorant_data)

@app.route('/chart')
def chart():
    # Fetch data from MySQL for Elo Gain
    cur_elo_gain = mysql.connection.cursor()
    cur_elo_gain.execute("SELECT `Elo-Gain` FROM valorant")
    elo_gain_data = cur_elo_gain.fetchall()
    cur_elo_gain.close()

    # Fetch data from MySQL for Mushroom class distribution
    cur_class_distribution = mysql.connection.cursor()
    cur_class_distribution.execute("SELECT `class` FROM mushrooms")
    class_distribution_data = cur_class_distribution.fetchall()
    cur_class_distribution.close()

    # Convert data to DataFrames
    df_elo_gain = pd.DataFrame(elo_gain_data, columns=['Elo Gain'])
    df_class_distribution = pd.DataFrame(class_distribution_data, columns=['class'])

    # Plotting Histogram for Elo Gain
    plt.figure(figsize=(8, 6))
    df_elo_gain.hist('Elo Gain')
    plt.xlabel('Elo Gains')
    plt.ylabel('Count')
    plt.title('Distribution of Elo Gains')
    histogram_img = BytesIO()
    plt.savefig(histogram_img, format='png')
    histogram_img.seek(0)
    plt.clf()

    # Plotting Pie Chart for Mushroom class distribution
    class_counts = df_class_distribution['class'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=['Edible (e)', 'Poisonous (p)'], colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    plt.title('Mushroom Distribution of Classes')
    pie_chart_img = BytesIO()
    plt.savefig(pie_chart_img, format='png')
    pie_chart_img.seek(0)

    # Pass the images as base64-encoded strings to the template
    histogram_encoded = base64.b64encode(histogram_img.read()).decode('utf-8')
    pie_chart_encoded = base64.b64encode(pie_chart_img.read()).decode('utf-8')

    return render_template('chart.html', histogram_img=histogram_encoded, pie_chart_img=pie_chart_encoded)

if __name__ == '__main__':
    app.run(debug=True)
