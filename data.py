import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Configurations
NUM_ROWS = 30_000_000  # Adjust this as needed
CHUNK_SIZE = 5_000_000  # Number of rows per chunk
CSV_FILE = "large_dataset_5GB.csv"
PARQUET_FOLDER = "parquet_chunks"  # Folder to store parquet chunks

# Ensure output folder exists
os.makedirs(PARQUET_FOLDER, exist_ok=True)

# Generate random data
def generate_data(rows):
    np.random.seed(42)  # Ensures reproducibility
    start_date = datetime(2020, 1, 1)

    data = {
        "id": np.arange(rows),
        "name": np.random.choice(["Alice", "Bob", "Charlie", "David", "Eve"], rows),
        "age": np.random.randint(18, 65, size=rows),
        "salary": np.random.uniform(30_000, 120_000, size=rows).round(2),
        "signup_date": [start_date + timedelta(days=np.random.randint(0, 3650)) for _ in range(rows)],
        "is_active": np.random.choice([True, False], size=rows),
    }

    df = pd.DataFrame(data)

    # Convert timestamps to microseconds (Spark-compatible)
    df["signup_date"] = df["signup_date"].astype("datetime64[us]")

    return df

# Write data in chunks
def write_data():
    for i in range(NUM_ROWS // CHUNK_SIZE):
        df_chunk = generate_data(CHUNK_SIZE)
        
        # Append CSV
        df_chunk.to_csv(CSV_FILE, mode="a", header=(i == 0), index=False)

        # Save each chunk as a separate Parquet file
        parquet_chunk_file = os.path.join(PARQUET_FOLDER, f"chunk_{i}.parquet")
        df_chunk.to_parquet(parquet_chunk_file, engine="pyarrow", index=False, compression="snappy")

        print(f"Chunk {i + 1} written: {parquet_chunk_file}")

# Run the script
if __name__ == "__main__":
    print("Generating dataset...")
    write_data()
    print("Dataset generation complete!")
