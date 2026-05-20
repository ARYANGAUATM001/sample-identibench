# System Identification Benchmark using IdentiBench

This project benchmarks multiple Mamba-based architectures for nonlinear system identification using the IdentiBench framework.

The goal is to compare training efficiency, prediction accuracy, stability, and generalization performance across different Mamba variants.

---

# 📌 Overview

We compare the following models:

- Mamba1
- Mamba2
- Mamba3

These models are evaluated on standard nonlinear system identification benchmarks using repeated experiments.

---

# ⚙️ Setup

## 1. Clone the repository

```bash
git clone https://github.com/ARYANGAUATM001/sample-identibench.git
cd sample-identibench

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Benchmark

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

### 🔹 1. Mamba1

1. Lightweight baseline implementation
2. Fast training
3. Stable across repetitions
4. Efficient for simpler dynamics



### 🔹 2. Mamba2

1. Improved optimization and representation
2. Better accuracy on nonlinear systems
3. Lower benchmark error
4. Faster inference


###  3. Mamba3

1. Larger and more expressive variant
2. Higher computational cost
3. Stronger modeling capability
4. More complex dynamics handling

---

## 📊 Results and Comparison

Each benchmark is repeated **2 times** to ensure reliable results.
Metrics are reported as **mean ± standard deviation**.

---



## 📊 Results

### 🔹 Mamba Results
![Mamba Results](utils/output1.png)

### 🔹 Mamba2 Results
![Mamba2 Results](utils/output2.png)

###  Mamba3
![Mamba3 Results](utils/output3.png)


## 🆚 Model Comparison

### 🔹 1. Prediction Accuracy

* **LSTM**

  * Achieves lower RMSE on most nonlinear benchmarks
  * Captures temporal and nonlinear relationships effectively

* **SSM**

  * Performs well on simpler dynamics
  * Limited capacity for highly nonlinear systems

---

### 🔹 2. Stability (Across Runs)

* **SSM**

  * Lower standard deviation
  * More stable and consistent

* **LSTM**

  * Slightly higher variance due to stochastic training

---

### 🔹 3. Training Time

* **SSM**

  * Faster training
  * Lower computational overhead

* **LSTM**

  * Slower due to sequential processing
  * More parameters to optimize

---

### 🔹 4. Generalization

* **LSTM**

  * Better generalization on unseen sequences
  * Handles long-term dependencies

* **SSM**

  * May underfit complex systems

---

## 📈 Key Takeaways

* There is a clear trade-off:

  * **SSM → efficiency and simplicity**
  * **LSTM → accuracy and flexibility**

* For **nonlinear system identification tasks**:

  * LSTM is generally more suitable

* For **resource-constrained or fast applications**:

  * SSM is a strong baseline

---

## 🧠 Interpretation of Results

* Lower **metric_score (mean)** → better performance
* Lower **std** → more stable model
* NaN values in some metrics may occur when:

  * a benchmark does not include that metric
  * the model struggles with extrapolation

---

## 📁 Project Structure

```
sample-identibench/
│
├── model/
│   ├── dss.py              # SSM implementation
│   ├── lstm.py             # LSTM model
│   ├── lstm_wrapper.py     
│   └── trainer.py          
│
├── utils/
│   ├── preprocessing.py
│   └── seed.py
│
├── configs.py              
├── main.py                 
├── README.md
└── requirements.txt
```



## 🎯 Conclusion

This project demonstrates how different model architectures behave on system identification benchmarks.

* **LSTM** improves predictive performance for complex nonlinear systems
* **SSM** provides a fast and stable baseline

The choice of model depends on:

* system complexity
* required accuracy
* computational constraints




