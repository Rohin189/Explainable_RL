import gymnasium as gym
import minigrid
import numpy as np
import pickle
import os
from stable_baselines3 import DQN

os.makedirs("./figures", exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)

class FlatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space["image"].shape
        flat_size = obs_shape[0] * obs_shape[1] * obs_shape[2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(flat_size,), dtype="float32"
        )
    def observation(self, obs):
        return obs["image"].flatten().astype("float32")

model = DQN.load("./checkpoints/best_model")
print("Model loaded.")

def run_clean_episodes(num):
    """Deterministic agent — all should succeed."""
    env = gym.make("MiniGrid-Empty-5x5-v0")
    wrapped = FlatObsWrapper(env)
    episodes = []

    for ep in range(num):
        obs, _ = wrapped.reset()
        episode = {"states": [], "actions": [], "rewards": [], "success": False}
        done = False

        while not done:
            episode["states"].append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = wrapped.step(int(action))
            episode["actions"].append(int(action))
            episode["rewards"].append(float(reward))
            done = terminated or truncated

        episode["success"] = sum(episode["rewards"]) > 0
        print(f"  [CLEAN] Episode {ep+1:02d} | Steps: {len(episode['actions']):3d} | Success: {episode['success']}")
        episodes.append(episode)

    wrapped.close()
    return episodes


def run_forced_failure_episodes(num, max_steps=3):
    """
    Force failures by capping episodes at max_steps=3.
    The goal requires at least 4–5 steps minimum, so truncating
    at 3 guarantees the agent never reaches it.
    """
    env = gym.make("MiniGrid-Empty-5x5-v0")
    wrapped = FlatObsWrapper(env)
    episodes = []

    for ep in range(num):
        obs, _ = wrapped.reset()
        episode = {"states": [], "actions": [], "rewards": [], "success": False}
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            episode["states"].append(obs.copy())
            # Mix of noisy + deterministic actions
            if np.random.rand() < 0.5:
                action = wrapped.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            episode["actions"].append(int(action))
            episode["rewards"].append(float(reward))
            step_count += 1
            done = terminated or truncated

        # Force mark as failure — episode was cut before completion
        episode["success"] = sum(episode["rewards"]) > 0
        print(f"  [FAIL]  Episode {ep+1:02d} | Steps: {len(episode['actions']):3d} | Success: {episode['success']}")
        episodes.append(episode)

    wrapped.close()
    return episodes
def run_natural_failure_episodes(num):
    """
    Let the agent act stochastically (deterministic=False).
    Some episodes will naturally fail due to exploration noise.
    """
    env = gym.make("MiniGrid-Empty-5x5-v0")
    wrapped = FlatObsWrapper(env)
    episodes = []

    for ep in range(num):
        obs, _ = wrapped.reset()
        episode = {"states": [], "actions": [], "rewards": [], "success": False}
        done = False

        while not done:
            episode["states"].append(obs.copy())

            # 🔥 KEY CHANGE: stochastic policy
            action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, _ = wrapped.step(int(action))
            episode["actions"].append(int(action))
            episode["rewards"].append(float(reward))

            done = terminated or truncated

        episode["success"] = sum(episode["rewards"]) > 0

        print(f"  [NATURAL] Episode {ep+1:02d} | Steps: {len(episode['actions']):3d} | Success: {episode['success']}")
        episodes.append(episode)

    wrapped.close()
    return episodes




print("\n--- Running 30 clean (success) episodes ---")
success_eps = run_clean_episodes(30)

print("\n--- Running 20 forced failure episodes ---")
forced_failure_eps = run_forced_failure_episodes(20, max_steps=3)

print("\n--- Running 20 natural episodes (mixed success/failure) ---")
natural_eps = run_natural_failure_episodes(20)

# 🔥 Combine all
trajectories = success_eps + forced_failure_eps + natural_eps

total_success = sum(t["success"] for t in trajectories)
total_failure = len(trajectories) - total_success

print(f"\n{'='*40}")
print(f"Total episodes : {len(trajectories)}")
print(f"Successes      : {total_success}")
print(f"Failures       : {total_failure}")
print(f"{'='*40}")

if total_failure < 5:
    print("WARNING: Too few failures. Reduce max_steps to 2 and rerun.")
else:
    with open("./checkpoints/trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    print("Trajectories saved to ./checkpoints/trajectories.pkl")

