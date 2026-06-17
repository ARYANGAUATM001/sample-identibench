# System Identification Benchmark using IdentiBench

This project benchmarks multiple Mamba-based architectures for nonlinear system identification using the IdentiBench framework.

The goal is to compare training efficiency, prediction accuracy, stability, and generalization performance across different Mamba variants.

---



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





# 🆚 Model Comparison

## 1. Prediction Accuracy

### Mamba2

- Achieved the best overall benchmark performance
- Lower RMSE across nonlinear system benchmarks


### Mamba

- Reliable baseline performance

---

## 

---





---

# 🧠 Interpretation of Results

- Lower `metric_score` indicates better benchmark performance
- Lower RMSE values correspond to improved prediction quality


NaN values may appear because:

- Certain datasets do not compute specific metrics
  

---

# 🎯 Conclusion

This project demonstrates the effectiveness of Mamba-based architectures for nonlinear system identification using the IdentiBench framework.


The ideal model choice depends on:

- available computational resources
- required prediction accuracy
- target system complexity


