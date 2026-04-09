import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from stable_baselines3 import DQN
import os

os.makedirs("./figures", exist_ok=True)

# --- Load model and trajectories ---
model = DQN.load("./checkpoints/best_model")
policy = model.policy
policy.set_training_mode(False)

with open("./checkpoints/trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

successes = [t for t in trajectories if t["success"]]
failures  = [t for t in trajectories if not t["success"]]

print(f"Successes: {len(successes)} | Failures: {len(failures)}")

# --- Compute saliency for a single state ---
def compute_saliency(state_flat):
    """
    Gradient of max Q-value w.r.t. input observation.
    Returns saliency as (7, 7, 3) array.
    """
    x = torch.tensor(state_flat, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)

    q_values = policy.q_net(x)
    max_q = q_values.max()
    max_q.backward()

    saliency = x.grad.data.abs().squeeze(0).numpy()
    return saliency.reshape(7, 7, 3)

# --- Aggregate saliency across an episode ---
def episode_mean_saliency(episode):
    maps = [compute_saliency(s) for s in episode["states"]]
    return np.mean(maps, axis=0)

# --- Compute mean saliency for success and failure groups ---
print("Computing saliency for success episodes...")
success_saliency = np.mean([episode_mean_saliency(e) for e in successes[:10]], axis=0)

print("Computing saliency for failure episodes...")
failure_saliency = np.mean([episode_mean_saliency(e) for e in failures[:10]], axis=0)

# Collapse across channels for visualization
success_map = success_saliency.mean(axis=2)
failure_map = failure_saliency.mean(axis=2)

# Normalize to [0, 1]
def normalize(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-8)

success_map = normalize(success_map)
failure_map = normalize(failure_map)
diff_map    = normalize(success_map - failure_map)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Saliency Map Analysis — DQN on MiniGrid-Empty-5x5", fontsize=13, fontweight="bold")

im0 = axes[0].imshow(success_map, cmap="YlOrRd", vmin=0, vmax=1)
axes[0].set_title("Success episodes\n(mean saliency)", fontsize=11)
axes[0].set_xlabel("Grid column")
axes[0].set_ylabel("Grid row")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(failure_map, cmap="YlOrRd", vmin=0, vmax=1)
axes[1].set_title("Failure episodes\n(mean saliency)", fontsize=11)
axes[1].set_xlabel("Grid column")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(diff_map, cmap="RdBu", vmin=0, vmax=1)
axes[2].set_title("Difference\n(success − failure)", fontsize=11)
axes[2].set_xlabel("Grid column")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

# Grid lines on all subplots
for ax in axes:
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig("./figures/saliency_maps.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ./figures/saliency_maps.png")

# --- Per-step saliency for one success episode ---
ep = successes[0]
fig2, axes2 = plt.subplots(1, len(ep["states"]), figsize=(3 * len(ep["states"]), 3))
fig2.suptitle("Per-step saliency — single success episode", fontsize=11)

if len(ep["states"]) == 1:
    axes2 = [axes2]

action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}

for i, state in enumerate(ep["states"]):
    sal = compute_saliency(state).mean(axis=2)
    sal = normalize(sal)
    axes2[i].imshow(sal, cmap="YlOrRd", vmin=0, vmax=1)
    axes2[i].set_title(f"Step {i+1}\nAction: {action_names.get(ep['actions'][i], '?')}", fontsize=9)
    axes2[i].axis("off")

plt.tight_layout()
plt.savefig("./figures/saliency_per_step.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ./figures/saliency_per_step.png")