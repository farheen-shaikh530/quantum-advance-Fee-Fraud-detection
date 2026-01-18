# classical_models.py
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_lr(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t1 = time.perf_counter()

    sec_per_pred = (t1 - t0) / max(len(X_test), 1)

    out = _metrics(y_test, y_pred)
    out.update({"sec_per_pred": sec_per_pred, "model": model})
    return out


def train_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t1 = time.perf_counter()

    sec_per_pred = (t1 - t0) / max(len(X_test), 1)

    out = _metrics(y_test, y_pred)
    out.update({"sec_per_pred": sec_per_pred, "model": model})
    return out


def train_mlp(X_train, y_train, X_test, y_test, seed=42):
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t1 = time.perf_counter()

    sec_per_pred = (t1 - t0) / max(len(X_test), 1)

    out = _metrics(y_test, y_pred)
    out.update({"sec_per_pred": sec_per_pred, "model": model})
    return out