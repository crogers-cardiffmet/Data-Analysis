import pandas as pd
import joblib
from sklearn.model_selection    import train_test_split, RandomizedSearchCV
from sklearn.ensemble           import RandomForestRegressor
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import mean_squared_error, r2_score
import numpy as np
from data_loader import load_data

df = load_data()
ext_ints = [
        c for c in df.columns
        if pd.api.types.is_integer_dtype(df[c].dtype) and df[c].isnull().any()
    ]
    for col in ext_ints:
        df[col] = df[col].astype("float64")
        
features = ['PM10','SO2','NO2','CO','O3']
X = df[features].fillna(df[features].median())
y = df['PM2.5'].fillna(df['PM2.5'].median())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3. Hyperparameter tuning
param_dist = {
    'n_estimators':    [100],
    'max_depth':       [20],
    'min_samples_split':[2],
    'min_samples_leaf':[4],
    'max_features':    ['sqrt']
}
base_rf = RandomForestRegressor(random_state=42)
search  = RandomizedSearchCV(
    base_rf, param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42
)
search.fit(X_train_s, y_train)

best_rf = search.best_estimator_
y_pred = best_rf.predict(X_test_s)

# 5. Persist
joblib.dump(best_rf,    "rf_model.pkl")
joblib.dump(scaler,     "scaler.pkl")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

s3 = boto3.client("s3")
BUCKET = "rf-model-cardiff"  

for local_path in [MODEL_PATH, SCALER_PATH, XTEST_PATH, YTEST_PATH]:
    s3.upload_file(local_path, BUCKET, local_path)
    print(f"Uploaded {local_path} → s3://{BUCKET}/{local_path}")
