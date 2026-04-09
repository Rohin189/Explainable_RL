import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
obs, _ = env.reset()
print("Observation shape:", obs["image"].shape)  # Should be (7, 7, 3)

for _ in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
print("Environment test passed.")