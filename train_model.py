from pathlib import Path
import json, time, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "dataset" / "source" / "split" / "images"
ANN_DIR = ROOT / "dataset" / "source" / "split" / "annotations"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (32, 32)
CAP_CLASS_ID = 0  # adjust only if the annotation class mapping in your dataset differs

def annotation_has_cap(annotation_path):
    if not annotation_path.exists():
        return False
    text = annotation_path.read_text(errors="ignore").strip()
    if not text:
        return False
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and int(parts[0]) == CAP_CLASS_ID:
            return True
    return False

def collect_data():
    if not IMG_DIR.exists():
        raise FileNotFoundError(
            f"{IMG_DIR} not found. Run download_dataset.py first."
        )
    X, y, names = [], [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in IMG_DIR.rglob("*") if p.suffix.lower() in extensions])
    if not files:
        raise RuntimeError("No images found in dataset/source/split/images.")

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        X.append(img.flatten().astype(np.float32) / 255.0)
        ann_matches = list(ANN_DIR.rglob(p.stem + ".txt"))
        ann = ann_matches[0] if ann_matches else None   
        y.append(1 if annotation_has_cap(ann) else 0)
        names.append(p.name)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Only one class was detected. Check CAP_CLASS_ID and annotation files."
        )
    return X, y, names

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "report": classification_report(
            y_test, pred, target_names=["Cap Off", "Cap On"], zero_division=0
        ),
        "cm": confusion_matrix(y_test, pred).tolist()
    }

def main():
    start = time.time()
    X, y, names = collect_data()
    print(f"Loaded {len(X)} images")
    print("Class counts:", {"Cap Off": int((y==0).sum()), "Cap On": int((y==1).sum())})
    np.savez_compressed(OUT / "prepared_data.npz", X=X, y=y)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM Linear": Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1))
        ]),
        "SVM RBF": Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1, gamma="scale"))
        ])
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        print(name, "5-fold CV:", scores, "mean:", scores.mean())
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # Hyperparameter tuning
    grid_model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC())
    ])
    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma": ["scale", "auto"]
    }
    search = GridSearchCV(
        grid_model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    best_eval = evaluate("Tuned SVM (GridSearchCV)", best, X_test, y_test)
    best_eval["best_params"] = search.best_params_
    best_eval["best_cv_score"] = float(search.best_score_)
    results.append(best_eval)
    joblib.dump(best, MODEL_DIR / "best_model.joblib")

    # Confusion matrix for best model
    pred = best.predict(X_test)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, pred),
        display_labels=["Cap Off", "Cap On"]
    )
    disp.plot()
    plt.title("Confusion Matrix - Tuned SVM")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    # Accuracy comparison
    names_plot = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    plt.figure(figsize=(8,4.5))
    plt.bar(names_plot, accs)
    plt.ylim(0,1.05)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "accuracy_comparison.png", dpi=180)
    plt.close()

    payload = {
        "dataset_images": int(len(X)),
        "cap_on": int((y==1).sum()),
        "cap_off": int((y==0).sum()),
        "image_size": list(IMG_SIZE),
        "feature_count": int(X.shape[1]),
        "test_size": 0.20,
        "random_state": 42,
        "results": results,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "runtime_seconds": round(time.time() - start, 2)
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print("\nDONE")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
