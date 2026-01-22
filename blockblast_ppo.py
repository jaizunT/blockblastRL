import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from blockblast_agent import BlockBlastEnv, TRAY_COUNT, TRAY_PAD_SIZE, get_valid_move_mask


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
                "clutter": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "holes": spaces.Box(0, 64, shape=(1,), dtype=np.float32),
            }
        )

    def _pack_obs(self, state):
        board, trays, moves, lines, combo, clutter, holes = state
        trays = np.stack(trays).astype(np.float32)
        return {
            "board": board.astype(np.float32),
            "trays": trays,
            "moves": np.array([moves], dtype=np.float32),
            "lines": np.array([lines], dtype=np.float32),
            "combo": np.array([combo], dtype=np.float32),
            "clutter": np.array([clutter], dtype=np.float32),
            "holes": np.array([holes], dtype=np.float32),
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

    def action_masks(self):
        mask = get_valid_move_mask(self.env.board, self.env.trays)
        return mask.squeeze(0).cpu().numpy().astype(bool)


def train_ppo(
    total_timesteps=10_000,
    model_path="ppo_blockblast.zip",
    save_freq=1_000,
    save_dir="ppo_checkpoints",
    use_masking=False,
    resume_step=None,
):
    env = BlockBlastGymEnv()
    resume_path = None
    if resume_step is not None:
        resume_path = f"{save_dir}/rl_model_{resume_step}_steps.zip"
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_dir)
    if use_masking:
        try:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
        except ImportError as exc:
            raise ImportError(
                "Masking requires sb3-contrib. Install with: pip install sb3-contrib"
            ) from exc
        env = ActionMasker(env, lambda e: e.action_masks())
        if resume_path:
            model = MaskablePPO.load(resume_path, env=env)
        else:
            model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    else:
        if resume_path:
            model = PPO.load(resume_path, env=env)
        else:
            model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=resume_step is None,
    )
    model.save(model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--save-dir", default="ppo_checkpoints")
    parser.add_argument("--model-path", default="ppo_blockblast.zip")
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--resume-step", type=int, default=None)
    args = parser.parse_args()
    train_ppo(
        total_timesteps=args.timesteps,
        model_path=args.model_path,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        use_masking=args.masking,
        resume_step=args.resume_step,
    )
