# qnn_model.py
import time
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def build_qnn(n_qubits, n_layers=2, encoding="dense", shots=None):
    """
    encoding:
      - "basic": one feature per qubit using RY(pi*x)
      - "dense": pack multiple features per qubit using RY/RZ/RX blocks
    shots:
      - None  => analytic simulator (deterministic gradients, faster, ideal for training)
      - int   => shot-based simulator (noisier gradients, more hardware-like)
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        # --------- Data Encoding ----------
        if encoding == "basic":
            for q in range(n_qubits):
                qml.RY(np.pi * x[q], wires=q)

        elif encoding == "dense":
            d = len(x)
            for q in range(n_qubits):
                i1 = (3*q + 0) % d
                i2 = (3*q + 1) % d
                i3 = (3*q + 2) % d

                a1 = 2 * np.pi * x[i1]
                a2 = 2 * np.pi * x[i2]
                a3 = 2 * np.pi * x[i3]

                qml.RY(a1, wires=q)
                qml.RZ(a2, wires=q)
                qml.RX(a3, wires=q)
        else:
            raise ValueError("encoding must be 'basic' or 'dense'")

        # --------- Variational Layers ----------
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[l, q, 0], wires=q)
                qml.RZ(weights[l, q, 1], wires=q)

            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        return qml.expval(qml.PauliZ(0))

    return circuit

def train_qnn(
    X_train, y_train, X_test, y_test,
    n_qubits=6, n_layers=2,
    steps=300, lr=0.05,
    batch_size=16,
    encoding="dense",
    optimizer_name="adam",     # "adam" or "gd"
    shots=None,                # None recommended for training
    seed=42
):
    """
    Proper gradient-based training using PennyLane Optimizers.

    Returns metrics + loss_history (for loss-vs-steps plots).
    """
    rng = np.random.default_rng(seed)

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test  = np.asarray(X_test, dtype=float)
    y_test  = np.asarray(y_test, dtype=int)

    circuit = build_qnn(n_qubits, n_layers=n_layers, encoding=encoding, shots=shots)

    # weights shape: (layers, qubits, 2)
    weights = rng.normal(scale=0.1, size=(n_layers, n_qubits, 2))
    weights = qml.numpy.array(weights, requires_grad=True)

    # pick optimizer
    if optimizer_name.lower() == "adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_name.lower() in ["gd", "gradientdescent"]:
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'gd'")

    def forward_prob(x, w):
        # circuit output in [-1,1] -> probability in [0,1]
        z = circuit(x, w)
        return (z + 1) / 2

    def batch_loss(w, Xb, yb):
        # binary cross-entropy
        eps = 1e-9
        probs = qml.numpy.array([forward_prob(x, w) for x in Xb])
        probs = qml.numpy.clip(probs, eps, 1 - eps)
        yb = qml.numpy.array(yb)
        return -qml.numpy.mean(yb * qml.numpy.log(probs) + (1 - yb) * qml.numpy.log(1 - probs))

    n = len(X_train)
    if n == 0:
        raise ValueError("Empty training set after preprocessing.")

    # ✅ store loss each step for learning-curve plot
    loss_history = []

    # --------- Training loop ----------
    for step in range(steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb, yb = X_train[idx], y_train[idx]

        weights = opt.step(lambda w: batch_loss(w, Xb, yb), weights)

        # record current batch loss (for plotting)
        loss_val = float(batch_loss(weights, Xb, yb))
        loss_history.append(loss_val)

        # optional: print progress
        if (step + 1) % 50 == 0:
            print(f"[QNN] step {step+1:4d}/{steps}, batch loss={loss_val:.4f}")

    # --------- Inference timing ----------
    t0 = time.perf_counter()
    probs_test = np.array([float(forward_prob(x, weights)) for x in X_test])
    y_pred = (probs_test >= 0.5).astype(int)
    t1 = time.perf_counter()

    sec_per_pred = (t1 - t0) / max(len(X_test), 1)

    return {
        "weights": np.array(weights),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "sec_per_pred": sec_per_pred,
        "loss_history": loss_history
    }