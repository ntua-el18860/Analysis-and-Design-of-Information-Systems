# Analysis-and-Design-of-Information-Systems

##  Semester Project
This project was developed as part of the **Analysis and Design of Information Systems** course in the **9th semester** of the **School of Electrical and Computer Engineering** at the **National Technical University of Athens (NTUA)**.

---

## **Contributors**
- **Alexandros Papadakos**(03118860)
- **Georgios Vitzilaios**(03116672)

---

## 📝 **Project Description**
The goal of this project is to develop a **big data processing and machine learning system** using **Apache Spark, Ray, and Hadoop**. The system is designed to:
- **Process large-scale datasets** stored in **HDFS (Hadoop Distributed File System)**
- **Perform ETL (Extract, Transform, Load) operations** efficiently
- **Train and optimize machine learning models** for predictive analysis
- **Benchmark performance** of data transformations and model training

The machine learning pipeline leverages **Random Forest Regression** with **hyperparameter tuning** to predict salary based on dataset features. The ETL process cleans and preprocesses the data before feeding it into the model.

---

##  **Dataset**
For this project, we used a **synthetically generated dataset** stored in **Parquet format** in HDFS. The dataset includes:
- Employee details such as **ID, Name, Age, Salary, Signup Date, and Activity Status**
- **Randomly generated dates**, including **future years (e.g., 2027, 2028)** due to the randomization process.
- **Large data volume**, requiring **distributed processing**.

The dataset is stored in **Parquet format** due to:
- **Optimized read & write performance** for Spark
- **Columnar storage for efficient querying**
- **Compression benefits** for large-scale data


The data is generated using :
```bash
python3 data.py
```
After generating the data , we should upload them to our HDFS filesystem.

---

##  **Setup & Installation**
### **1 Prerequisites**
Ensure the following are installed:
- **Python 3.8+**
- **Apache Spark**
- **Hadoop (HDFS)**
- **Ray**
- **pipenv or virtualenv** (recommended)

### **2 Install Required Python Libraries**
Clone the repository and install dependencies:
```bash
git clone https://github.com/ntua-el18860/Analysis-and-Design-of-Information-Systems
```
### **3 Start HDFS & Spark Cluster**
Ensure Hadoop and Spark are running:
```bash
start-dfs.sh
start-yarn.sh
```
### **4 Running ETL jobs**
You can execute the ETL jobs with :
```bash
python3 etl_pipeline.py
python3 etl_query.py
python3 etl_query_2.py
```
The pipeline logs execution time, memory usage, and transformation details in: etl_benchmark.log

### **5 Running the ML Training Job**
First you need to start the **Ray cluster** with : 
**master** 
```bash
ray start --head --port=6379
```
**worker**
```bash
ray start --address='master:6379'
```
Then on the **master** you can run :
```bash
python3 ray_ml_pipeline.py
```
To make prediction you must run :
```bash
python3 prediction.py
```
You can read the predicted data on **predictions.csv**

