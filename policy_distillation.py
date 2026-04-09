import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

os.makedirs("./figures", exist_ok=True)

with open("./checkpoints/trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

# --- Build dataset from ALL episodes ---
all_states  = []
all_actions = []

for ep in trajectories:
    all_states.extend(ep["states"])
    all_actions.extend(ep["actions"])

X = np.array(all_states)   # shape: (N, 147)
y = np.array(all_actions)  # shape: (N,)

print(f"Dataset size : {X.shape[0]} state-action pairs")
print(f"Action distribution: { {a: int((y==a).sum()) for a in np.unique(y)} }")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit decision tree (depth 4 keeps it readable) ---
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree.predict(X_train))
test_acc  = accuracy_score(y_test,  tree.predict(X_test))
print(f"\nDecision tree fidelity:")
print(f"  Train accuracy : {train_acc:.3f}")
print(f"  Test  accuracy : {test_acc:.3f}")

# --- Text rules ---
action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
rules = export_text(tree, feature_names=[f"obs_{i}" for i in range(147)])
with open("./figures/tree_rules.txt", "w") as f:
    f.write(f"Decision Tree Fidelity — Train: {train_acc:.3f} | Test: {test_acc:.3f}\n\n")
    f.write(rules)
print("Saved: ./figures/tree_rules.txt")

# --- Plot tree ---
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    tree,
    max_depth=4,
    feature_names=[f"obs_{i}" for i in range(147)],
    class_names=action_names,
    filled=True,
    rounded=True,
    impurity=False,
    proportion=False,
    fontsize=8,
    ax=ax
)
ax.set_title(
    f"Policy distillation — Decision tree surrogate (depth 4)\n"
    f"Train fidelity: {train_acc:.3f}  |  Test fidelity: {test_acc:.3f}",
    fontsize=12
)
plt.tight_layout()
plt.savefig("./figures/policy_tree.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ./figures/policy_tree.png")

# --- Feature importance (top 15) ---
importances = tree.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(range(15), importances[top_idx], color="#1D9E75")
ax2.set_xticks(range(15))
ax2.set_xticklabels([f"obs_{i}" for i in top_idx], rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Feature importance")
ax2.set_title("Top 15 most important observation features\n(policy distillation tree)", fontsize=11)
plt.tight_layout()
plt.savefig("./figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ./figures/feature_importance.png")