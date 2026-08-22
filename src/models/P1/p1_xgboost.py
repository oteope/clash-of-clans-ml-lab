import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from mlflow_tracking.tracking_utils import (
    configure_tracking,
    mlflow_run,
    log_dataset_context,
    log_split_config,
    log_model_params,
    log_metrics,
    log_model_and_artifacts,
)
from mlflow_tracking.experiments import get_experiment_name
from sklearn.preprocessing import LabelEncoder

# Load the dataset
print("[1/7] Loading dataset...")
DATASET_PATH = "data/datasets/role_classification.parquet"
data = pd.read_parquet(DATASET_PATH)

# Define features and target variable
print("[2/7] Preparing features and target...")
X = data.drop(columns=[
    "player_tag",
    "clan_tag",
    "war_frequency",
    "war_league",
    "capital_league",
    "type",
    "is_family_friendly",
    "role"
])
y = data['role']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
print("[3/7] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Mlflow experiment
print("[4/7] Starting MLflow run...")
experiment_name = get_experiment_name("p1")

with mlflow_run(
    experiment_name,
    run_name = "Xgboost baseline",
):
    
    #Loading the dataset
    print("[5/7] Logging dataset and split configuration...")
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
        target = "role",
    )

    #Split configuration
    log_split_config(
        "train_test_split",
        0.2,
        42,
        {},
    )
    
    #Model
    print("[6/7] Training and evaluating model...")
    model = xgb.XGBClassifier(n_estimators=100,
                              random_state=42) 
    
    #Model hyperparameters
    log_model_params(
       {
        "n_estimators":100,
        "random_state":42,
       }
    )
    
    #Model training
    model.fit(X_train,y_train)
    
    #Prediction
    y_pred = model.predict(X_test)
    
    # Convert predictions back to original labels
    y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))
    
    #Metrics
    metrics = {
        "accuracy": accuracy_score(y_test,y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        }
    
    #Save metrics
    log_metrics(metrics)
    
    #Save model
    log_model_and_artifacts(model)
    
    print("[7/7] MLflow run completed successfully!")

print("✓ Training pipeline finished.")