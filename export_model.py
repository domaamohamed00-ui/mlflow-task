import shutil
import mlflow.xgboost
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import xgboost as xgb

print("Preparing California Housing data...")
california = fetch_california_housing()
X, y = california.data, california.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تدريب النموذج الأفضل (Run 2: depth=5, lr=0.05)
print("Training optimal model (Run 2)...")
params = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)

# مسح المجلد القديم إن وجد ثم حفظ الموديل بصيغة MLflow
shutil.rmtree("./model", ignore_errors=True)
mlflow.xgboost.save_model(xgb_model=model, path="./model")

print(
    "SUCCESS: Model saved directly to ./model directory for FastAPI serving!"
)