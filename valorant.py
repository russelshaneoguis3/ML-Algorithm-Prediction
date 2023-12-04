
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import pickle

# Load the dataset
filename = 'valorant.csv'
dataframe = pd.read_csv(filename)

# Separate features (X) and target variable (y)
X = dataframe.drop('Elo-Gain', axis=1)  # Exclude 'Elo Gain' as a feature
y = dataframe['Elo-Gain']

# Perform train/test split (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72)

# Initialize the Elastic Net model for regression
alpha = 0.1  # Regularization strength
l1_ratio = 0.1  # Mixing parameter, 0 for L2 penalty, 1 for L1 penalty

model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=72)

# Train the model
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('valorant_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
