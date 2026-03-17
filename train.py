# train.py

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             roc_curve)

# ── 1. LOAD PREPROCESSED DATA ─────────────────────────────────
X_train = pickle.load(open('data/X_train.pkl', 'rb'))
X_test  = pickle.load(open('data/X_test.pkl',  'rb'))
y_train = pickle.load(open('data/y_train.pkl', 'rb'))
y_test  = pickle.load(open('data/y_test.pkl',  'rb'))

print("✅ Data loaded!")

# ── 2. DEFINE MODELS ───────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes":         MultinomialNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                   class_weight='balanced',
                                                   random_state=42)
}

# ── 3. TRAIN & EVALUATE ALL MODELS ────────────────────────────
os.makedirs('plots', exist_ok=True)
os.makedirs('data',  exist_ok=True)

results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'Accuracy':  acc,
        'Precision': prec,
        'Recall':    rec,
        'F1 Score':  f1,
        'ROC-AUC':   auc
    }

    print(f"  ✅ Accuracy:  {acc:.4f}")
    print(f"  ✅ Precision: {prec:.4f}")
    print(f"  ✅ Recall:    {rec:.4f}")
    print(f"  ✅ F1 Score:  {f1:.4f}")
    print(f"  ✅ ROC-AUC:   {auc:.4f}")

# ── 4. CONFUSION MATRICES ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=150)
plt.show()
print("\n✅ Saved: plots/confusion_matrices.png")

# ── 5. ROC CURVES ─────────────────────────────────────────────
plt.figure(figsize=(8, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})", color=color, lw=2)

plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=150)
plt.show()
print("✅ Saved: plots/roc_curves.png")

# ── 6. MODEL COMPARISON BAR CHART ─────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
model_names = list(results.keys())

scores = {m: [results[name][m] for name in model_names] for m in metrics}

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=name)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0.8, 1.0)
ax.set_ylabel('Score')
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150)
plt.show()
print("✅ Saved: plots/model_comparison.png")

# ── 7. PICK BEST MODEL & SAVE IT ──────────────────────────────
best_name = max(results, key=lambda n: results[n]['F1 Score'])
best_model = results[best_name]['model']

pickle.dump(best_model, open('data/best_model.pkl', 'wb'))

print(f"\n🏆 Best Model: {best_name}")
print(f"   F1 Score: {results[best_name]['F1 Score']:.4f}")
print(f"   ROC-AUC:  {results[best_name]['ROC-AUC']:.4f}")
print("\n✅ Saved: data/best_model.pkl")
print("\n🎉 Phase 3 Complete! Ready for Phase 4 — Deployment!")
