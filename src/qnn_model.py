# qnn_model.py
import time
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def build_qnn(n_qubits: int, n_layers: int = 2, encoding: str = "dense", shots=None):
    """
    Build a PennyLane QNode for a QNN.

    encoding:
      - "basic": one feature per qubit using RY(pi*x[q])
      - "dense": pack multiple features per qubit using RY/RZ/RX blocks
    shots:
      - None  => analytic simulator (deterministic gradients)
      - int   => shot-based simulator (noisier, hardware-like)
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        # --------- Data Encoding ----------
        if encoding == "basic":
            # assumes len(x) >= n_qubits
            for q in range(n_qubits):
                qml.RY(np.pi * x[q], wires=q)

        elif encoding == "dense":
            d = len(x)
            for q in range(n_qubits):
                i1 = (3 * q + 0) % d
                i2 = (3 * q + 1) % d
                i3 = (3 * q + 2) % d

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

            # linear entanglement
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        # single-qubit readout
        return qml.expval(qml.PauliZ(0))

    return circuit


def train_qnn(
    X_train, y_train, X_test, y_test,
    n_qubits: int = 6,
    n_layers: int = 2,
    steps: int = 300,
    lr: float = 0.05,
    batch_size: int = 16,
    encoding: str = "dense",
    optimizer_name: str = "adam",
    shots=None,
    seed: int = 42
):
    """
    Gradient-based training for QNN using PennyLane optimizers.

    Returns:
      dict with keys:
        accuracy, precision, recall, f1, sec_per_pred, loss_history
    """
    rng = np.random.default_rng(seed)

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)

    if len(X_train) == 0:
        raise ValueError("Empty training set after preprocessing.")

    circuit = build_qnn(n_qubits=n_qubits, n_layers=n_layers, encoding=encoding, shots=shots)

    # weights shape: (layers, qubits, 2)
    w0 = rng.normal(scale=0.1, size=(n_layers, n_qubits, 2))
    weights = qml.numpy.array(w0, requires_grad=True)

    # optimizer selection
    if optimizer_name.lower() == "adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_name.lower() in ["gd", "gradientdescent", "sgd"]:
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'gd'")

    def forward_prob(x, w):
        # map circuit output z in [-1,1] -> probability in [0,1]
        z = circuit(x, w)
        return (z + 1.0) / 2.0

    def batch_loss(w, Xb, yb):
        # binary cross entropy
        eps = 1e-9
        probs = qml.numpy.array([forward_prob(x, w) for x in Xb])
        probs = qml.numpy.clip(probs, eps, 1 - eps)
        yb = qml.numpy.array(yb)
        return -qml.numpy.mean(yb * qml.numpy.log(probs) + (1 - yb) * qml.numpy.log(1 - probs))

    n = len(X_train)
    loss_history = []

    # --------- Training loop ----------
    for step in range(steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb, yb = X_train[idx], y_train[idx]

        weights = opt.step(lambda w: batch_loss(w, Xb, yb), weights)

        loss_val = float(batch_loss(weights, Xb, yb))
        loss_history.append(loss_val)

        if (step + 1) % 50 == 0:
            print(f"[QNN] step {step+1:4d}/{steps}, batch loss={loss_val:.4f}")

    # --------- Inference timing ----------
    t0 = time.perf_counter()
    probs_test = np.array([float(forward_prob(x, weights)) for x in X_test])
    y_pred = (probs_test >= 0.5).astype(int)
    t1 = time.perf_counter()

    sec_per_pred = (t1 - t0) / max(len(X_test), 1)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "sec_per_pred": sec_per_pred,
        "loss_history": loss_history,
    }