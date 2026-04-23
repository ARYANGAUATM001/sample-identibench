# Nonlinear System Identification using IdentiBench

##  Overview

This project applies machine learning techniques to **nonlinear system identification** using the IdentiBench benchmarking framework. The goal is to model system dynamics from input-output time-series data and evaluate performance on standardized tasks.

---

## About IdentiBench

IdentiBench is a benchmarking framework for system identification that provides:

* Standard datasets (e.g., cascaded tanks)
* Unified training and evaluation pipeline
* Consistent performance metrics

This allows fair comparison of different models under identical conditions.

---

##  Model Implementation

### 🔹 Selective State Space Model (SSM)

* Custom-built model
* Uses **input-dependent state transitions**
* Inspired by modern architectures such as Mamba

This model aims to dynamically control how past information influences future predictions.

---

## 🎯 Objective

* Learn nonlinear system dynamics
* Predict system outputs from sequential inputs
* Evaluate model performance using standardized benchmarks

---

## 📂 Project Structure

```
project/
│
├── model/                 
│   └── dss.py            # Selective State Space Model
│
├── trainer.py            # Training logic
├── main.py               # Runs IdentiBench experiments
├── results/              # Output CSV and logs
└── README.md
```

---

## 🚀 How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Run benchmark

```
python main.py
```

---

## 📊 Results and Evaluation



![Output](utils/output.png)

The model was evaluated across multiple IdentiBench tasks. The output metrics summarize training time, inference time, and prediction accuracy.

### 🔹 Key Results (Example Output)

* **metric_score:** ~45.6
* **cs_multisine_rmse:** ~49.3
* **cs_arrow_full_rmse:** ~48.5
* **cs_arrow_no_extrapolation_rmse:** ~38.97

---



###  Overall Performance

* The **metric_score (~45.6)** indicates moderate prediction accuracy
* The model successfully captures key system dynamics

---

###  Dataset-wise Behavior

* **Multisine & Arrow datasets (~48–49 RMSE)**
  → Moderate error suggests the model learns general patterns but struggles with complex variations

* **No-extrapolation case (~38.97 RMSE)**
  → Significantly lower error
  → Indicates strong performance when predicting within known data distribution

---



### ✔ Stability

* All benchmark runs completed successfully
* No training instability observed

---

### ✔ NaN Values

* Some metrics appear as `NaN`
* This is expected in IdentiBench for certain configurations
* It does **not indicate failure**

---



---

## Key Takeaways

* Selective SSM provides **adaptive temporal modeling**
* Model performs better on **seen data than extrapolated scenarios**
* IdentiBench enables **consistent and fair evaluation**
  

---

##  References

* IdentiBench Benchmark Framework
* PyTorch Documentation
  

---

## 👨‍💻 Author

* Your Name
