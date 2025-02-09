from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
import time
import psutil  # Ensure this is installed in ray_env
import os

# ✅ Initialize Spark Session
spark = SparkSession.builder \
    .appName("ETL Benchmarking") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ✅ Define log file
LOG_FILE = "etl_benchmark.log"

# ✅ Start time measurement
start_time = time.time()

# ✅ Track memory usage before loading data
memory_before = psutil.virtual_memory().used / (1024 ** 3)  # GB

# ✅ Read all Parquet chunks from the HDFS directory
print("⏳ Loading Parquet chunks from HDFS...")
parquet_path = "hdfs:///datasets/parquet_chunks/"  # Change path here
df = spark.read.parquet(parquet_path)

# ✅ Track memory usage after loading data
memory_after = psutil.virtual_memory().used / (1024 ** 3)  # GB

# ✅ Log dataset info
print(f"✅ Loaded dataset with {df.count()} rows and {len(df.columns)} columns.")

# ✅ Show schema for debugging
df.printSchema()

# ✅ Perform ETL operations
print("🔄 Running ETL transformations...")

df_transformed = df \
    .withColumnRenamed("name", "full_name") \
    .withColumn("signup_date", to_timestamp(col("signup_date"), "yyyy-MM-dd HH:mm:ss"))

# ✅ Save transformed data back to HDFS
output_parquet_path = "hdfs:///datasets/processed_large_dataset.parquet"
df_transformed.write.mode("overwrite").parquet(output_parquet_path)

# ✅ Measure execution time
execution_time = time.time() - start_time

# ✅ Write benchmark log
with open(LOG_FILE, "a") as log_file:
    log_file.write("🚀 ETL Process Benchmark\n")
    log_file.write(f"🔹 Rows Processed: {df.count()}\n")
    log_file.write(f"🔹 Columns: {len(df.columns)}\n")
    log_file.write(f"⏳ Execution Time: {execution_time:.2f} seconds\n")
    log_file.write(f"💾 Memory Usage Before: {memory_before:.2f} GB\n")
    log_file.write(f"💾 Memory Usage After: {memory_after:.2f} GB\n")
    log_file.write("=" * 40 + "\n")

print(f"✅ ETL process completed in {execution_time:.2f} seconds.")
print(f"📂 Processed dataset saved at: {output_parquet_path}")
print(f"📜 Benchmark log saved in: {LOG_FILE}")

# ✅ Stop Spark session
spark.stop()
