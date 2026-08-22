import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
DATASET_PATH = "data/dataset/role_classification.parquet"
data = pd.read_parquet(DATASET_PATH)

# Define features and target variable
X = data.drop(columns=['role'])
Y = data['role']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
