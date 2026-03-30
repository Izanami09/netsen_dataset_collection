"""
NetSentinel - App Fingerprinting Model Training
Trains 1D-CNN and XGBoost to identify apps from encrypted traffic patterns.

Inspired by PACKETPRINT (NDSS 2022) - uses packet size sequences,
inter-arrival times, and flow statistics for app classification.

Usage:
    python train_app_fingerprint.py
    
Prerequisites:
    - Collected app traffic using collect_app_traffic.py
    - Data should be in data/app_traffic/*.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Dense, Dropout,
        Flatten, BatchNormalization, Input, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. Only XGBoost will be trained.")

warnings.filterwarnings("ignore")

# ─── Configuration ───
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "app_traffic")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Packet Size Sequence length (from PACKETPRINT)
PSS_LENGTH = 50


def load_data():
    """Load all collected app traffic CSVs."""
    print("=" * 60)
    print("📂 Loading App Traffic Data")
    print("=" * 60)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    if not csv_files:
        print(f"\n❌ No data found in {DATA_DIR}")
        print("   Run 'sudo python collect_app_traffic.py' first to collect data.")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        df = pd.read_csv(path)
        print(f"  Loaded: {f} ({len(df)} flows)")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total flows: {len(combined):,}")
    print(f"  Apps found: {combined['app'].nunique()}")
    print(f"  Distribution:\n{combined['app'].value_counts().to_string()}")

    return combined


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrices for training."""
    print("\n" + "=" * 60)
    print("🔧 Preparing Features")
    print("=" * 60)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["app"])
    app_names = le.classes_
    n_classes = len(app_names)

    print(f"  Classes: {list(app_names)}")
    print(f"  Number of classes: {n_classes}")

    # ── Feature Set 1: Statistical features (for XGBoost) ──
    stat_features = [
        "flow_duration", "total_packets", "total_bytes",
        "packets_per_sec", "bytes_per_sec",
        "pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max",
        "pkt_size_median", "pkt_size_q25", "pkt_size_q75", "pkt_size_skew",
        "payload_mean", "payload_std", "payload_max", "payload_zero_ratio",
        "iat_mean", "iat_std", "iat_min", "iat_max", "iat_median",
        "upload_packets", "download_packets",
        "upload_ratio", "download_ratio", "up_down_ratio",
        "up_bytes_total", "up_size_mean", "up_size_std", "up_size_max",
        "down_bytes_total", "down_size_mean", "down_size_std", "down_size_max",
        "up_iat_mean", "up_iat_std", "down_iat_mean", "down_iat_std",
        "n_bursts", "avg_burst_size", "max_burst_size",
        "is_tcp", "is_udp",
    ]

    available_stat = [f for f in stat_features if f in df.columns]
    X_stat = df[available_stat].fillna(0).replace([np.inf, -np.inf], 0)

    # Scale
    scaler = StandardScaler()
    X_stat_scaled = pd.DataFrame(
        scaler.fit_transform(X_stat),
        columns=X_stat.columns
    )

    print(f"  Statistical features: {len(available_stat)}")

    # ── Feature Set 2: Packet Size Sequence (for 1D-CNN) ──
    pss_cols = [f"pss_{i}" for i in range(PSS_LENGTH)]
    available_pss = [c for c in pss_cols if c in df.columns]

    if available_pss:
        X_pss = df[available_pss].fillna(0).values
        # Normalize PSS
        pss_max = np.max(np.abs(X_pss)) + 1e-8
        X_pss_norm = X_pss / pss_max
        print(f"  PSS features: {len(available_pss)} (sequence length)")
    else:
        X_pss_norm = None
        print("  ⚠️ No PSS features found")

    # Save preprocessing artifacts
    joblib.dump(le, os.path.join(MODEL_DIR, "app_label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "app_scaler.pkl"))
    joblib.dump(available_stat, os.path.join(MODEL_DIR, "app_stat_features.pkl"))
    joblib.dump(pss_max, os.path.join(MODEL_DIR, "app_pss_max.pkl"))

    return X_stat_scaled, X_pss_norm, y, app_names, available_stat


def train_xgboost(X_train, X_test, y_train, y_test, app_names):
    """Train XGBoost on statistical features."""
    print("\n" + "=" * 60)
    print("🚀 Training XGBoost (Statistical Features)")
    print("=" * 60)

    n_classes = len(app_names)
    import time
    start = time.time()

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    xgb.fit(X_train, y_train)
    train_time = time.time() - start

    pred = xgb.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Time:      {train_time:.2f}s")
    print(f"\n{classification_report(y_test, pred, target_names=app_names)}")

    joblib.dump(xgb, os.path.join(MODEL_DIR, "app_xgboost.pkl"))
    
    return xgb, pred, acc, f1, train_time


def train_cnn(X_pss_train, X_pss_test, y_train, y_test, app_names):
    """Train 1D-CNN on Packet Size Sequences."""
    if not TF_AVAILABLE:
        print("  ⚠️ Skipping CNN (TensorFlow not available)")
        return None, None, 0, 0, 0

    print("\n" + "=" * 60)
    print("🧠 Training 1D-CNN (Packet Size Sequences)")
    print("=" * 60)

    n_classes = len(app_names)
    import time
    start = time.time()

    # Reshape for CNN: (samples, sequence_length, 1)
    X_train_cnn = X_pss_train.reshape(X_pss_train.shape[0], X_pss_train.shape[1], 1)
    X_test_cnn = X_pss_test.reshape(X_pss_test.shape[0], X_pss_test.shape[1], 1)

    model = Sequential([
        Input(shape=(X_train_cnn.shape[1], 1)),

        # Block 1 — learn local packet patterns
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 2 — learn medium-range burst patterns
        Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 3 — learn long-range traffic patterns
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.3),

        # Classification
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1,
    )

    train_time = time.time() - start

    pred_proba = model.predict(X_test_cnn)
    pred = np.argmax(pred_proba, axis=1)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Time:      {train_time:.2f}s")
    print(f"\n{classification_report(y_test, pred, target_names=app_names)}")

    model.save(os.path.join(MODEL_DIR, "app_cnn_model.keras"))

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("1D-CNN Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("1D-CNN Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "app_cnn_history.png"), dpi=150)
    plt.close()

    return model, pred, acc, f1, train_time


def generate_plots(y_test, xgb_pred, cnn_pred, app_names, xgb_results, cnn_results):
    """Generate evaluation plots."""
    print("\n  📊 Generating plots...")

    # Confusion matrix — XGBoost
    fig, axes = plt.subplots(1, 2 if cnn_pred is not None else 1,
                              figsize=(20 if cnn_pred is not None else 10, 8))
    if cnn_pred is None:
        axes = [axes]

    cm = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=app_names, yticklabels=app_names)
    axes[0].set_title(f"XGBoost (F1: {xgb_results['f1']:.3f})", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    if cnn_pred is not None:
        cm2 = confusion_matrix(y_test, cnn_pred)
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                    xticklabels=app_names, yticklabels=app_names)
        axes[1].set_title(f"1D-CNN (F1: {cnn_results['f1']:.3f})", fontweight="bold")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

    plt.suptitle("App Fingerprinting — Confusion Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "app_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Per-app F1 scores
    from sklearn.metrics import f1_score as f1_per
    xgb_per_app = f1_per(y_test, xgb_pred, average=None)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(app_names))
    width = 0.35

    ax.bar(x - width/2, xgb_per_app, width, label="XGBoost", color="#2196F3")
    if cnn_pred is not None:
        cnn_per_app = f1_per(y_test, cnn_pred, average=None)
        ax.bar(x + width/2, cnn_per_app, width, label="1D-CNN", color="#4CAF50")

    ax.set_xlabel("App")
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-App F1 Scores", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(app_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "app_per_app_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("  ✅ Plots saved")


# ─── Main ───
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║  🛡️ NetSentinel — App Fingerprinting Training     ║
    ║  Identify apps from encrypted traffic patterns    ║
    ╚══════════════════════════════════════════════════╝
    """)

    df = load_data()
    X_stat, X_pss, y, app_names, stat_features = prepare_features(df)

    # Split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

    X_stat_train, X_stat_test = X_stat.iloc[train_idx], X_stat.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train XGBoost
    xgb_model, xgb_pred, xgb_acc, xgb_f1, xgb_time = train_xgboost(
        X_stat_train, X_stat_test, y_train, y_test, app_names
    )

    xgb_results = {"accuracy": xgb_acc, "f1": xgb_f1, "time": xgb_time}

    # Train 1D-CNN
    cnn_pred = None
    cnn_results = {"accuracy": 0, "f1": 0, "time": 0}
    if X_pss is not None:
        X_pss_train, X_pss_test = X_pss[train_idx], X_pss[test_idx]
        cnn_model, cnn_pred, cnn_acc, cnn_f1, cnn_time = train_cnn(
            X_pss_train, X_pss_test, y_train, y_test, app_names
        )
        cnn_results = {"accuracy": cnn_acc, "f1": cnn_f1, "time": cnn_time}

    # Generate plots
    generate_plots(y_test, xgb_pred, cnn_pred, app_names, xgb_results, cnn_results)

    # Summary
    print("\n" + "=" * 60)
    print("✅ APP FINGERPRINTING TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  XGBoost:  Accuracy={xgb_acc:.4f}  F1={xgb_f1:.4f}  Time={xgb_time:.1f}s")
    if cnn_pred is not None:
        print(f"  1D-CNN:   Accuracy={cnn_results['accuracy']:.4f}  F1={cnn_results['f1']:.4f}  Time={cnn_results['time']:.1f}s")
    print(f"\n  Models saved to: {MODEL_DIR}/")
    print(f"  Plots saved to: {REPORT_DIR}/")
    print(f"\n  Next: Integrate into dashboard or collect more data.")
