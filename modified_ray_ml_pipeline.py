import ray
import joblib
import logging
import pandas as pd
import numpy as np
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Set up logging
logging.basicConfig(filename="ml_pipeline.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize Spark
spark = SparkSession.builder.appName("Ray_ML_Pipeline").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")  # Fix Arrow issue
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")  # Disable Arrow fallback

# Load Parquet Data
parquet_path = "/datasets/parquet_chunks/"
print(f"🔄 Searching for Parquet files in HDFS `{parquet_path}`...")
df = spark.read.parquet(parquet_path)
print(f"📊 Available columns: {df.columns}")

# Select Features and Target
features = ['age', 'is_active']  # Additional feature added
target = 'salary'
print(f"✅ Selected Features: {features}")
print(f"🎯 Target Column: {target}")

df_sample = df.select(*features, target).limit(20000)  # Increase dataset size for better model performance

# ✅ Convert Spark DataFrame to Pandas without Arrow
try:
    print("🔄 Converting Spark DataFrame to Pandas...")
    collected_data = df_sample.collect()  # Collect into Python list
    pandas_df = pd.DataFrame([row.asDict() for row in collected_data])  # Convert to Pandas
    print("✅ Successfully converted Spark DataFrame to Pandas")
except Exception as e:
    print(f"❌ Error converting to Pandas: {e}")
    logging.error(f"❌ Error converting to Pandas: {e}")
    exit()

# Handle missing values
pandas_df.dropna(inplace=True)

# Split dataset
X = pandas_df[features]
y = pandas_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42)

# Track Training Time
start_time = time.time()
random_search.fit(X_train, y_train)
training_time = time.time() - start_time

# Best Model
best_model = random_search.best_estimator_

# Evaluate Model
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"📉 Optimized Model Training Complete! \nMean Squared Error: {mse}\nR² Score: {r2}\nMean Absolute Error: {mae}\nTraining Time: {training_time:.2f} seconds")
logging.info(f"📉 Optimized Model Training Complete! MSE: {mse}, R²: {r2}, MAE: {mae}, Training Time: {training_time:.2f}s")

# Save Model
joblib.dump(best_model, "ml_model.pkl")
logging.info("✅ Model saved as 'ml_model.pkl'")
print("✅ Model saved as 'ml_model.pkl'")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as 'scaler.pkl'")
