import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
filename = 'mushrooms.csv'
dataframe = pd.read_csv(filename)

# Separate features (X) and target variable (y)
X = dataframe.drop('class', axis=1)
y = dataframe['class']

# Convert categorical variables into numerical format using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the dataset into training and testing sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model with the correct path
model_path = 'mushroom_classifier_model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)