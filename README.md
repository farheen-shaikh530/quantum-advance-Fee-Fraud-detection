<p align="center">
  <img src="assets/qnn-hero.gif" alt="Quantum Fraud Detection Hero Banner" width="100%" />
</p>

<h1 align="center">
   Quantum vs Classical Models for Rapid Advance-Fee Fraud Detection under NISQ Constraints
</h1>

<p align="center">
  <b>Accepted and Presented at IEEE SoutheastCon 2026</b><br>
 <b>Author: Farheen Shabbir Shaikh</b><br>
 <b>Publication: March 2026</b><br>
</p>


---

## 🚨 Situation Analysis: Advance-Fee Fraud (AFF)

A message appears:

> “Loan approved instantly — just pay a $1000 processing fee.”

It feels real. It feels urgent.

The user pays.

And then—silence.

 **Reality:**
- No lender  
- No loan  
- No approval
 

**Victims:**
- Individuals with low credit scores, struggling to secure loans   
- Job seekers in urgent financial need  
- Victims of relationship-based scams  

 ~~A classic **Advance-Fee Fraud (AFF)** attack — designed to exploit urgency and trust before the victim has time to question it.

---

## 💡 Motivation

- **7,097 AFF complaints (IC3 Report 2024)**
-  Victims pay upfront for **non-existent services**
-  Fraud happens **within minutes → requires early detection**

- Traditional ML models rely on historical patterns. This work focuses on **early behavioral signals**

---

## 🧠 Proposed Approach

We propose a **Quantum Neural Network (QNN)** framework that:

- Classify **early-stage fraud signals**
- Uses **behavior-driven temporal features**
- Operates under **NISQ (Noisy Intermediate-Scale Quantum) constraints**
- Benchmarks against **3 widely used ML models**

---

## 🗂️ Custom: Dataset Construction

| Feature | Description |
|--------|------------|
| `bank_account_no` | User identifier |
| `scam_msg_time` | Timestamp of scam message |
| `first_pay_time` | Timestamp of first payment |
| `amount` | Transaction amount |
| `Flag` | 1 = Fraud, 0 = Non-Fraud |


### Data Distribution is imbalance
- Total Records: **2,000**
- Fraud Cases: **100 (5%)**
- Non-Fraud: **1,900 (95%)**

⚠️ **Challenge:** Imbalanced dataset → risk of bias toward non-fraud class

---

## ⚙️ Feature Engineering(Novel)

Inter-Payment Interval (IPI): Behavioral Feature Engineering: Transform raw timestamps and amounts into meaningful signals. 50 seconds < IPI = higher likelihood of fraud

---

## 🧠 Model Architecture

### Quantum Model
- Variational Quantum Neural Network (QNN)
- Angle encoding
- Qubit configurations: **4, 5, 6, 7**
- Shallow circuits (NISQ-friendly)
- Entanglement: CNOT layers

### Classical Baselines
- Logistic Regression  
- Support Vector Machine (RBF)  
- Neural Network (MLP)


---

## ⚙️ Training Strategy

- Mini-batch training  
- Adam optimizer  
- Loss monitoring  

---

## 📊 Experimental Results


<h3 align="center"> Result 1: Performance vs Qubit Count</h2>

<p align="center">
<img width="700" height="380" alt="Screenshot 2026-03-22 at 12 26 59 AM" src="https://github.com/user-attachments/assets/87be3008-7cfb-4d81-8d8d-6c35af5ce727" />
</p>

<h3 align="center"> Result 2: Model Comparison </h2>
<p align="center">
<img width="700" height="380" alt="Screenshot 2026-03-22 at 12 27 10 AM" src="https://github.com/user-attachments/assets/5752c929-ffa6-471c-b0e8-dd2c33c32ca6" />
</p>

<h3 align="center"> Result 3: Training Loss (On qubit scaling)</h2>

<p align="center">

<img src="https://github.com/user-attachments/assets/b1ba4b6f-590f-4a58-9474-53cd704eadfb" width="400" height="220"/>
<img src="https://github.com/user-attachments/assets/f3178bb9-5995-4e0c-a969-8d1173d3208e" width="400" height="220"/>
<img src="https://github.com/user-attachments/assets/7e363a41-6b72-4fa9-ba35-563ef7c8bd05" width="400" height="220"/>
<img src="https://github.com/user-attachments/assets/71fff49a-3582-4998-a12a-27a6ba27ce7f" width="400" height="220"/>

</p>


---

## 🔬 Observations

-  QNN achieves competitive performance against SVM
-  It underperforms against classical Neural Networks due to: Limited qubits (NISQ) and  PennyLane constraints
-  Underperform in Qubits > 5 case: When feeding input to quantum gates, superposition increased instability due to its multi-states within one state nature
  - Pennylane simulator has noisy device (default.mixed) plugin that cause loss of quantum information hence **Decoherence** creates training data loss.

---

## 🏆 Best Model Performance (5-Qubit QNN)

| Metric | Value |
|-------|------|
| Accuracy | **0.820** |
| Precision | **0.858** |
| Recall | **0.793** |
| F1-score | **0.824** |
| Inference Latency | **2.92 × 10⁻³ sec** |

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall 
- F1-score  
- Training loss curves  
- Qubit scaling impact  
- Inference latency  

---

### ⚛️ Future Extensions
- Scale simulations to **higher qubits (100–500+)** and trying new simulator
- Improve **noise-resilient training** through optimzation algorithm
- Stabilize gradients in VQCs
- Add **ROC-AUC & PR-AUC** metrics evaluation
- Explore adaptive threshold set for comparing IPI feature value.

---
