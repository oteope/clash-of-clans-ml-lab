import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split 

#Loading the dataset
DATASET_PATH = "data/dataset/role_classification.parquet"
data = pd.read_parquet(DATASET_PATH)

#Defining features and target variable
X = data.drop(columns=['role'])
y = data['role']

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Making predictions on the test set
y_pred = model.predict(X_test)