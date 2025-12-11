import pandas as pd
from train_svm_lr import train_logistic_regression, train_svm
from train_qnn import train_qnn

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_excel("transactions.csv")

X = df[['ipi_minutes']].values
y = df['label'].values

print("Dataset loaded. Samples:", len(X))

# ---------------------------
# 2. Run Logistic Regression
# ---------------------------
print("\nRunning Logistic Regression...")
lr_model, lr_acc, lr_cm = train_logistic_regression(X, y)
print("Logistic Regression Accuracy:", lr_acc)
print("Confusion Matrix:\n", lr_cm)

# ---------------------------
# 3. Run SVM
# ---------------------------
print("\nRunning Support Vector Machine...")
svm_model, svm_acc, svm_cm = train_svm(X, y)
print("SVM Accuracy:", svm_acc)
print("Confusion Matrix:\n", svm_cm)

# ---------------------------
# 4. Run Quantum Neural Network (QNN)
# ---------------------------
print("\nRunning Quantum Neural Network...")
qnn_model, qnn_acc, qnn_cm = train_qnn(X, y)
print("QNN Accuracy:", qnn_acc)
print("Confusion Matrix:\n", qnn_cm)

print("\nAll models trained successfully.")
