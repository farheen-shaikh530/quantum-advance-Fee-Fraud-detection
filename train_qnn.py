# -------------------------------------------------------------
# train_qnn.py (IPI ONLY VERSION)
# -------------------------------------------------------------

import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def train_qnn(X, y, epochs=40):
    X = np.array(X, requires_grad=False)
    y = np.array(y, requires_grad=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Normalize IPI to [0,1]
    X_min = np.min(X_train)
    X_max = np.max(X_train)
    X_train_norm = (X_train - X_min) / (X_max - X_min)
    X_test_norm = (X_test - X_min) / (X_max - X_min)

    # 1 feature → 1 qubit
    n_qubits = 1
    dev = qml.device("default.qubit", wires=n_qubits)

    def feature_encoding(x):
        qml.RX(x[0] * np.pi, wires=0)

    def variational_block(w):
        qml.RY(w[0], wires=0)

    @qml.qnode(dev)
    def qnn(x, w):
        feature_encoding(x)
        variational_block(w)
        return qml.expval(qml.PauliZ(0))

    # weight initialization
    weights = np.random.uniform(0, np.pi, 1, requires_grad=True)

    def mse_loss(w, Xb, yb):
        preds = [(qnn(x, w) + 1) / 2 for x in Xb]
        preds = np.array(preds)
        return np.mean((preds - yb) ** 2)

    opt = qml.GradientDescentOptimizer(0.2)

    # Training
    for epoch in range(epochs):
        weights, loss = opt.step_and_cost(
            lambda w: mse_loss(w, X_train_norm, y_train),
            weights
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss={loss}")

    # Predict
    def predict(Xn):
        preds = [(qnn(x, weights) + 1) / 2 for x in Xn]
        preds = np.array(preds)
        return (preds > 0.5).astype(int)

    y_pred = predict(X_test_norm)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    qnn_model = {"weights": weights, "X_min": X_min, "X_max": X_max}

    return qnn_model, accuracy, cm