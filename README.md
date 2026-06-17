# System Identification Benchmark using IdentiBench

This project benchmarks multiple Mamba-based architectures for nonlinear system identification using the IdentiBench framework.

The goal is to compare training efficiency, prediction accuracy, stability, and generalization performance across different Mamba variants.

---

# 📌 Overview

We compare the following models:

- Mamba
- Mamba2
- Mamba3

These models are evaluated on standard nonlinear system identification benchmarks using repeated experiments.

---
# ⚙️ Setup

## 1. Clone the repository

```bash
git clone https://github.com/ARYANGAUATM001/sample-identibench.git
cd sample-identibench
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Benchmark

Run the main script:

```bash
python main.py --model mamba1
python main.py --model mamba2
python main.py --model mamba3
```

This will:

1. Train the selected model
2. Run IdentiBench benchmarks
3. Repeat experiments multiple times
4. Output evaluation metrics

---

## 🧠 Models Implemented

### 🔹 1. Mamba

1. Lightweight baseline implementation




### 🔹 2. Mamba2

1. Improved optimization and representation



###  3. Mamba3

1. Larger and more expressive variant


---





## 📊 Results

### 🔹 Mamba Results
![Mamba Results](utils/output1.png)

### 🔹 Mamba2 Results
![Mamba2 Results](utils/output2.png)

###  Mamba3
![Mamba3 Results](utils/output3.png)




# 🆚 Model Comparison

## 1. Prediction Accuracy

### Mamba2

- Achieved the best overall benchmark performance
- Lower RMSE across nonlinear system benchmarks


### Mamba

- Reliable baseline performance


### Mamba3

- Higher model capacity


---

## 2. Stability Across Runs

### Mamba1

- Most stable training behavior
- Lower variance across repeated experiments
- Consistent benchmark outputs

### Mamba2

- Good repeatability
- Stable convergence during training

### Mamba3

- Higher variance due to increased model complexity
- More sensitive to hyperparameter settings

---

## 3. Training Efficiency

### Mamba2

- Fastest benchmark execution
- Lower inference overhead
- Efficient training-performance tradeoff

### Mamba1

- Moderate computational cost
- Good efficiency for baseline experiments

### Mamba3

- Highest computational overhead
- Longer training duration
- Increased resource consumption

---





---

# 🧠 Interpretation of Results

- Lower `metric_score` indicates better benchmark performance
- Lower RMSE values correspond to improved prediction quality
- Smaller standard deviation indicates more stable training behavior

NaN values may appear because:

- Certain datasets do not compute specific metrics
- Some extrapolation benchmarks may not apply to all runs

---

# 🎯 Conclusion

This project demonstrates the effectiveness of Mamba-based architectures for nonlinear system identification using the IdentiBench framework.

Key observations:

- Mamba2 produced the strongest practical benchmark performance
- Mamba1 serves as an efficient lightweight baseline
- Mamba3 offers higher modeling capacity but requires additional optimization

The ideal model choice depends on:

- available computational resources
- required prediction accuracy
- target system complexity


