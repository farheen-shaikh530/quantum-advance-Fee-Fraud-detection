### Paper Title : Quantum vs Classical Models for Rapid Advance-Fee Fraud Detection Under NISQ Constraints


## Publication & Conference 

This work has been **accepted for publication** at an IEEE-sponsored international conference.

**Author:** Farheen Shabbir Shaikh 
**Conference:** IEEE SoutheastCon 2026  
**Organized by:** IEEE Region 3  
**Paper Status:** Accepted (to be published)  
**Publication Date:** **March 13, 2026**  
**Conference Website:** https://ieeesoutheastcon.org/call-for-papers/

**Overview:** The paper presents an experimental evaluation of Quantum Neural Networks (QNNs) for time-critical Advance-Fee Fraud (AFF) detection, emphasizing optimization behavior, qubit scaling effects, and comparison against classical machine-learning baselines.

This repository implements and evaluates **Quantum Neural Networks (QNNs)** against **classical machine-learning models** for detecting **Advance-Fee Fraud (AFF)** using compact behavioral features. The primary focus is on **Inter-Payment Interval (IPI)**, a time-critical feature that captures attacker-induced urgency. The project investigates whether **Quantum Machine Learning (QML)** can provide competitive detection performance under **Noisy Intermediate-Scale Quantum (NISQ)** constraints, and how **optimization behavior and qubit scaling** affect model reliability.

---

## 📌 Project Objectives

- To design a Quantum Machine Learning framework for Advance-Fee Fraud detection using behavior-driven temporal features.
- To empirically analyze Quantum Neural Network (QNN) performance under Noisy Intermediate-Scale Quantum (NISQ) constraints, with a focus on noise sensitivity, shallow circuit design, and optimization stability.
- To evaluate the impact of qubit scaling on training convergence, loss behavior, and detection accuracy in variational quantum circuits.
- To investigate how quantum feature encoding and optimization strategies influence inference latency and detection reliability.
- To compare QNN performance against classical machine-learning baselines using identical feature sets and evaluation metrics.
 

---

## 🧠 Models Implemented

### Quantum Model
- **Quantum Neural Network (QNN)**
- Variational Quantum Circuit (VQC)
- Dense angle encoding
- Qubit configurations: **4, 5, 6, 7**
- Simulator-based execution (NISQ emulation)

###  Classical Baselines
- **Logistic Regression**
- **Support Vector Machine (RBF kernel)**

---

##  Features Used

Behavior-driven features designed for early-stage fraud detection:

- **Inter-Payment Interval (IPI)** – primary feature
- Response time between scam messages
- Estimated message count prior to payment
- Log-transformed transaction amount
- Account-level z-score normalization (amount & IPI)

---

## ⚙️ Tech Stack

### Programming & Libraries
- **Python 3.9+**
- **PennyLane** – quantum ML framework
- **NumPy** – numerical computing
- **Pandas** – data processing
- **scikit-learn** – classical ML models & preprocessing
- **Matplotlib** – visualization

### Quantum Simulation
- PennyLane default.qubit simulator
- NISQ-aware shallow circuits

### Tooling
- Git & GitHub (version control)
- VS Code
- Virtual environments (venv)

---

## 📈 Evaluation Metrics

All models are evaluated using the same train/test splits:

- **Accuracy**
- **Precision**
- **Recall** *(critical)*
- **F1-score**
- **Training loss curves** (QNN optimization behavior)
- **Performance vs qubit count**
- **Inference latency**

---

## Installation & Setup

### ✅ macOS or Linux


## 1. Clone repository

``` bash
git clone https://github.com/farheen-shaikh530/quantum-advance-Fee-Fraud-detection.git
cd quantum-advance-Fee-Fraud-detection
```

## 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```



## 📊 Experimental Results

The following figures illustrate the **training dynamics of the Quantum Neural Network (QNN)** under simulated **Noisy Intermediate-Scale Quantum (NISQ)** conditions. These experiments analyze how **decoherence and noise influence optimization stability and convergence** in variational quantum circuits.

<p align="center">

<img src="https://github.com/user-attachments/assets/b1ba4b6f-590f-4a58-9474-53cd704eadfb" width="400" height="220"/>
<img src="https://github.com/user-attachments/assets/f3178bb9-5995-4e0c-a969-8d1173d3208e" width="400" height="220"/>

<br><br>

<img src="https://github.com/user-attachments/assets/7e363a41-6b72-4fa9-ba35-563ef7c8bd05" width="400" height="220"/>
<img src="https://github.com/user-attachments/assets/71fff49a-3582-4998-a12a-27a6ba27ce7f" width="400" height="220"/>

</p>

### 🔬 Result Interpretation

- **Figure 1:** Early-stage training loss during QNN optimization under simulated decoherence.
- **Figure 2:** Loss fluctuations reflecting noise sensitivity in variational circuits.
- **Figure 3:** Training convergence behavior across iterations.
- **Figure 4:** Stabilization phase of the QNN optimization process under NISQ constraints.

These results highlight how **quantum noise and decoherence affect the optimization landscape of variational quantum circuits**, emphasizing the importance of **noise-aware circuit design and robust optimization strategies for practical QML systems**.

---
