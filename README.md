<p align="center">
  <img src="assets/qnn-hero.gif" alt="Quantum Fraud Detection Hero Banner" width="100%" />
</p>

<p align="center">
  <img width="2560" height="1280" alt="quantum_cover" src="https://github.com/user-attachments/assets/646ed3d3-0a4f-4b44-9bed-d13741d04ef0" />
</p>

<p align="center">
  <b>Project Type:</b> Research Project <br>
  <b>Published Date:</b> April 20, 2026 <br>
  <b>Published Conference:</b> SoutheastCon 2026 <br>
  <b>DOI:</b> <a href="https://ieeexplore.ieee.org/document/11476481">https://ieeexplore.ieee.org/document/11476481</a> <br>
  <b>Skills:</b> Scikit-learn, PyTorch, Matplotlib, Pandas, NumPy, Quantum Neural Network, Logistic Regression
</p>

---

## Overview

The proposed **Quantum Neural Network (QNN)** framework:

- Classifies **early-stage fraud signals**
- Uses **behavior-driven temporal features**
- Operates under **NISQ (Noisy Intermediate-Scale Quantum) constraints**
- Benchmarks against **3 widely used ML models**

---

## Motivation

Social engineering attacks in financial institutions—such as Business Email Compromise (BEC), whaling, smishing, vishing, and scareware—have been widely studied, typically focusing on isolated attack instances. The aim of this study is to experiment how Quantum Machine Learning can perform rapid classifications for Advance-Fee Fraud (AFF). AFF is underexplored, particularly in how these attacks unfold in sequential phases that can lead to significant financial loss at the final phase. IC3 (2024) reports 7,097 AFF complaints, where victims are persuaded to make upfront payments for non-existent services, often within minutes—highlighting the need for early detection. Unlike traditional machine learning approaches that rely on historical patterns, this work emphasizes early behavioral signals to detect and intervene in fraud at its initial stages.

---

## Customized Dataset Construction

| Feature | Description |
|---------|-------------|
| `bank_account_no` | User identifier |
| `scam_msg_time` | Timestamp of scam message |
| `first_pay_time` | Timestamp of first payment |
| `amount` | Transaction amount |
| `Flag` | 1 = Fraud, 0 = Non-Fraud |

### Data Distribution (Imbalanced)

- **Total Records:** 2,000
- **Fraud Cases:** 100 (5%)
- **Non-Fraud:** 1,900 (95%)

> **Challenge:** Imbalanced dataset → risk of bias toward non-fraud class

---

## Feature Engineering (Novel)

**Inter-Payment Interval (IPI):** Behavioral feature engineering that transforms raw timestamps and amounts into meaningful signals. IPI < 50 seconds = higher likelihood of fraud.

---

## Model Architecture

<p align="center">
  <img width="1360" height="1520" alt="QNN Architecture" src="https://github.com/user-attachments/assets/0251b469-0afb-4765-b941-d6e380b16309" />
</p>

### Quantum Model

- Variational Quantum Neural Network (QNN)
- Angle encoding
- Qubit configurations: **4, 5, 6, 7**
- Shallow circuits (NISQ-friendly)
- Entanglement: CNOT layers

### Classical Baselines

- Logistic Regression
- Support Vector Machine
- Neural Network

---

## Training Strategy

- Mini-batch training
- Adam optimizer
- Loss monitoring

---

## Research Results

<p align="center">
  <img width="700" height="380" alt="A-PRF Performance" src="https://github.com/user-attachments/assets/87be3008-7cfb-4d81-8d8d-6c35af5ce727" />
</p>
<h5 align="center">Result 1: A-PRF Performance against Qubit count 4, 5, 6, 7</h5>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/b1ba4b6f-590f-4a58-9474-53cd704eadfb" width="400" height="220"/>
  <img src="https://github.com/user-attachments/assets/f3178bb9-5995-4e0c-a969-8d1173d3208e" width="400" height="220"/>
  <img src="https://github.com/user-attachments/assets/7e363a41-6b72-4fa9-ba35-563ef7c8bd05" width="400" height="220"/>
  <img src="https://github.com/user-attachments/assets/71fff49a-3582-4998-a12a-27a6ba27ce7f" width="400" height="220"/>
</p>
<h5 align="center">Result 2: Training Loss during Qubit Scaling</h5>

---

<p align="center">
  <img width="700" height="380" alt="Model Comparison" src="https://github.com/user-attachments/assets/5752c929-ffa6-471c-b0e8-dd2c33c32ca6" />
</p>
<h5 align="center">Result 3: Model Comparison — Logistic Regression, SVM, Neural Network, and Best Performing QNN (5 Qubits)</h5>

---

## Observations

- QNN achieves competitive performance against SVM.
- Underperforms against classical Neural Networks due to limited qubits (NISQ) and PennyLane constraints.
- Qubits > 5: When feeding input to quantum gates, superposition increased instability due to its multi-state nature.
- PennyLane simulator's noisy device (`default.mixed`) causes loss of quantum information, leading to **decoherence** and training data loss.

---

## Best Model Performance (5-Qubit QNN)

| Metric | Value |
|--------|-------|
| Accuracy | **0.820** |
| Precision | **0.858** |
| Recall | **0.793** |
| F1-Score | **0.824** |
| Inference Latency | **2.92 × 10⁻³ sec** |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Training loss curves
- Qubit scaling impact
- Inference latency

---

## Future Extensions

- Scale simulations to **higher qubits (100–500+)** using different simulators (e.g., cuQuantum, Qiskit).
- Improve **noise-resilient training** through optimization algorithms (e.g., QAOA, VQE).
- Add **ROC-AUC & PR-AUC** metrics for evaluating same attack type.
- Explore threshold tuning for IPI feature value to flag fraud (0 or 1).
