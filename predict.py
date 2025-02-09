import ray
import joblib
import logging
import pandas as pd
from pyspark.sql import SparkSession

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Set up logging
logging.basicConfig(filename="prediction.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize Spark
spark = SparkSession.builder.appName("Prediction_Pipeline").getOrCreate()

# 🔄 **Ensure Arrow is disabled to avoid conversion issues**
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# Load Model and Scaler
print("🔄 Loading Trained Model and Scaler...")
try:
    model = joblib.load("ml_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler Loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load New Data from HDFS
hdfs_input_path = "/datasets/parquet_chunks/"  # Modify if necessary
print(f"🔄 Reading New Data from HDFS: `{hdfs_input_path}`...")

df_new = spark.read.parquet(hdfs_input_path)
print(f"📊 Available columns: {df_new.columns}")

# Select only relevant features
features = ['age']
if 'name' in df_new.columns:
    features.append('name')  # Keep names for logging

df_sample = df_new.select(*features).limit(5000)  # ⚠️ Reduce data size to avoid memory issues

# 🔄 **Fix: Convert Spark DataFrame to Pandas safely**
try:
    pandas_df = pd.DataFrame(df_sample.collect(), columns=df_sample.columns)
except Exception as e:
    print(f"❌ Error converting to Pandas: {e}")
    exit(1)

# Handle missing values
pandas_df.dropna(inplace=True)

# Extract name column if available
name_column = None
if 'name' in pandas_df.columns:
    name_column = pandas_df.pop('name')  # Remove name from features but keep it for logging

# Scale features
X_new = scaler.transform(pandas_df)

# Make predictions
print("🔄 Making Predictions...")
predictions = model.predict(X_new)

# Create a DataFrame for logging and saving
results_df = pd.DataFrame({"Prediction": predictions})
if name_column is not None:
    results_df.insert(0, "Name", name_column)

# Save locally
results_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved locally as `predictions.csv`")

# Log predictions
for idx, row in results_df.iterrows():
    logging.info(f"Name: {row.get('Name', 'Unknown')}, Prediction: {row['Prediction']}")

# Try to save to HDFS
hdfs_output_path = "/datasets/output/predictions.csv"
try:
    results_df.to_csv(hdfs_output_path, index=False)
    print(f"✅ Predictions successfully saved to HDFS at `{hdfs_output_path}`")
except Exception as e:
    print(f"⚠️ Warning: Failed to save predictions to HDFS: {e}")
