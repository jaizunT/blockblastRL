import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from blockblast_agent import BlockBlastEnv, TRAY_COUNT, TRAY_PAD_SIZE


class BlockBlastGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = BlockBlastEnv()
        self.action_space = spaces.Discrete(3 * 8 * 8)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0, 1, shape=(8, 8), dtype=np.float32),
                "trays": spaces.Box(
                    0, 1, shape=(TRAY_COUNT, TRAY_PAD_SIZE, TRAY_PAD_SIZE), dtype=np.float32
                ),
                "moves": spaces.Box(0, 1000, shape=(1,), dtype=np.float32),
                "lines": spaces.Box(0, 8, shape=(1,), dtype=np.float32),
                "combo": spaces.Box(0, 1000, shape=(1,), dtype=np.float32),
            }
        )

    def _pack_obs(self, state):
        board, trays, moves, lines, combo = state
        trays = np.stack(trays).astype(np.float32)
        return {
            "board": board.astype(np.float32),
            "trays": trays,
            "moves": np.array([moves], dtype=np.float32),
            "lines": np.array([lines], dtype=np.float32),
            "combo": np.array([combo], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return self._pack_obs(state), {}

    def step(self, action):
        tray = action // 64
        rem = action % 64
        row = rem // 8
        col = rem % 8
        state, reward, done, info = self.env.step((tray, row, col))
        obs = self._pack_obs(state)
        return obs, float(reward), bool(done), False, info


def train_ppo(
    total_timesteps=10_000,
    model_path="ppo_blockblast.zip",
    save_freq=1_000,
    save_dir="ppo_checkpoints",
):
    env = BlockBlastGymEnv()
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_dir)
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(model_path)
    return model


if __name__ == "__main__":
    train_ppo()
