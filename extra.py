import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN
import os

os.makedirs("./figures", exist_ok=True)

# --- Load data ---
model = DQN.load("./checkpoints/best_model")
policy = model.policy
policy.set_training_mode(False)

with open("./checkpoints/trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

successes = [t for t in trajectories if t["success"]]
failures  = [t for t in trajectories if not t["success"]]

print(f"Success: {len(successes)} | Failure: {len(failures)}")


# ==============================
# 🔥 1. SUCCESS vs FAILURE BAR
# ==============================
success_count = len(successes)
failure_count = len(failures)

plt.figure()
plt.bar(["Success", "Failure"], [success_count, failure_count])
plt.title("Success vs Failure Episodes")
plt.ylabel("Number of Episodes")
plt.savefig("./figures/success_vs_failure.png", dpi=150)
plt.show()


# ==============================
# 🔥 2. SALIENCY OVER TIME
# ==============================
def compute_saliency(state_flat):
    x = torch.tensor(state_flat, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)

    q_values = policy.q_net(x)
    max_q = q_values.max()
    max_q.backward()

    saliency = x.grad.data.abs().squeeze(0).numpy()
    return saliency.mean()  # single value per step


# Take one success episode
ep = successes[0]

saliency_values = []

for state in ep["states"]:
    sal = compute_saliency(state)
    saliency_values.append(sal)

plt.figure()
plt.plot(saliency_values, marker='o')
plt.title("Saliency Over Time (Single Episode)")
plt.xlabel("Step")
plt.ylabel("Average Saliency")
plt.savefig("./figures/saliency_over_time.png", dpi=150)
plt.show()


# ==============================
# 🔥 3. FULL EPISODE HEATMAP
# ==============================
def compute_saliency_map(state_flat):
    x = torch.tensor(state_flat, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)

    q_values = policy.q_net(x)
    max_q = q_values.max()
    max_q.backward()

    saliency = x.grad.data.abs().squeeze(0).numpy()
    return saliency.reshape(7, 7, 3).mean(axis=2)


maps = [compute_saliency_map(s) for s in ep["states"]]
full_map = np.mean(maps, axis=0)

plt.figure()
plt.imshow(full_map)
plt.title("Full Episode Saliency Heatmap")
plt.colorbar()
plt.savefig("./figures/heatmap_full_episode.png", dpi=150)
plt.show()


# ==============================
# 🔥 4. SALIENCY BAR (TOP FEATURES)
# ==============================
# Flatten importance
flat_importance = full_map.flatten()
top_idx = np.argsort(flat_importance)[-10:]

plt.figure()
plt.bar(range(len(top_idx)), flat_importance[top_idx])
plt.xticks(range(len(top_idx)), top_idx)
plt.title("Top Salient Features")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.savefig("./figures/saliency_bar.png", dpi=150)
plt.show()


print("All extra visualizations saved in ./figures/")