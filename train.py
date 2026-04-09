import gymnasium as gym
import minigrid
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

ENV_ID = "MiniGrid-Empty-5x5-v0"
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

class RewardShapingWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 1.0 if reward > 0 else -0.01
        return obs, reward, terminated, truncated, info

def make_env():
    env = gym.make(ENV_ID, max_steps=200)
    env = RewardShapingWrapper(env)
    env = FlatObsWrapper(env)
    env = Monitor(env)
    return env

env = make_env()
eval_env = make_env()

checkpoint_cb = CheckpointCallback(
    save_freq=25_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="dqn_minigrid"
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=CHECKPOINT_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1
)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=500,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    train_freq=4,
    target_update_interval=500,
    verbose=1,
    tensorboard_log=LOG_DIR
)

print("Starting training...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, eval_cb]
)

model.save(os.path.join(CHECKPOINT_DIR, "dqn_final"))
print("Training complete. Model saved.")