import pandas as pd
import joblib
from sklearn.model_selection    import train_test_split, RandomizedSearchCV
from sklearn.ensemble           import RandomForestRegressor
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import mean_squared_error, r2_score
import numpy as np

@st.cache_data
def load_data(path="combined_output.csv"):
    df = pd.read_csv(path)
    if {'year','month','day','hour'}.issubset(df.columns):
        df['date'] = pd.to_datetime(
            df[['year','month','day','hour']].astype(str).agg('-'.join, axis=1),
            format='%Y-%m-%d-%H'
        )
    return df 

df = load_data()
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
