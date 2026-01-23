import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from blockblast_agent import BlockBlastEnv, TRAY_COUNT, TRAY_PAD_SIZE, get_valid_move_mask
from blockblast_simulation import BlockBlastSim, valid_placement


def _action_mask_fn(env):
    target = env
    if not hasattr(target, "action_masks"):
        if hasattr(target, "unwrapped") and hasattr(target.unwrapped, "action_masks"):
            target = target.unwrapped
        elif hasattr(target, "env") and hasattr(target.env, "action_masks"):
            target = target.env
    mask = target.action_masks() if hasattr(target, "action_masks") else None
    if mask is None:
        return np.ones(3 * 8 * 8, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.ones(3 * 8 * 8, dtype=bool)
    if mask.ndim == 2:
        row_any = mask.any(axis=1)
        if not np.all(row_any):
            if getattr(env, "_debug_mask", False):
                print("[mask] empty per-env mask row; forcing all-true for those envs")
            mask[~row_any] = True
        return mask
    if not mask.any():
        if getattr(env, "_debug_mask", False):
            print("[mask] empty mask; forcing all-true")
        mask = np.ones(3 * 8 * 8, dtype=bool)
    return mask


class EpisodeLogger(BaseCallback):
    def __init__(self, log_every=1000):
        super().__init__()
        self.log_every = log_every
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_count += 1
            if self.log_every > 0 and (self.episode_count % self.log_every == 0):
                print(
                    f"[ppo] episode={self.episode_count} "
                    f"reward={episode.get('r', 0):.2f} "
                    f"len={episode.get('l', 0)}",
                    flush=True,
                )
        return True


def _wrap_monitor(env):
    if isinstance(env, VecEnv):
        return VecMonitor(env)
    return Monitor(env)


class BlockBlastGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = BlockBlastEnv()
        self._no_valid_moves = False
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
        if self._no_valid_moves:
            self._no_valid_moves = False
            obs = self._pack_obs(self.env._encode_state())
            return obs, -7.5, True, False, {"terminal": "no_valid_moves"}
        tray = action // 64
        rem = action % 64
        row = rem // 8
        col = rem % 8
        state, reward, done, info = self.env.step((tray, row, col))
        obs = self._pack_obs(state)
        return obs, float(reward), bool(done), False, info

    def action_masks(self):
        mask = get_valid_move_mask(self.env.board, self.env.trays, force_nonempty=False)
        mask = mask.squeeze(0).cpu().numpy().astype(bool)
        if not mask.any():
            self._no_valid_moves = True
            mask[:] = False
            mask[0] = True
        return mask


class BlockBlastSimGymEnv(gym.Env):
    def __init__(self, use_model_batch=False):
        super().__init__()
        self.env = BlockBlastSim(use_model_batch=use_model_batch)
        self._no_valid_moves = False
        self._debug_mask = False
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

    def _pad_tray(self, tray, size=TRAY_PAD_SIZE):
        tray_arr = np.array(tray, dtype=np.float32)
        h, w = tray_arr.shape
        padded = np.zeros((size, size), dtype=np.float32)
        if h > size or w > size:
            tray_arr = tray_arr[:size, :size]
            h, w = tray_arr.shape
        padded[:h, :w] = tray_arr
        return padded

    def _pack_obs(self, state):
        trays = np.stack([self._pad_tray(tray) for tray in state["batch"]]).astype(np.float32)
        return {
            "board": state["board"].astype(np.float32),
            "trays": trays,
            "moves": np.array([state["moves_since_last_clear"]], dtype=np.float32),
            "lines": np.array([state["lines_cleared_last_move"]], dtype=np.float32),
            "combo": np.array([state["combo_count"]], dtype=np.float32),
            "clutter": np.array([state["clutter"]], dtype=np.float32),
            "holes": np.array([state["holes"]], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return self._pack_obs(state), {}

    def step(self, action):
        if self._no_valid_moves:
            self._no_valid_moves = False
            obs = self._pack_obs(self.env._get_state())
            return obs, -7.5, True, False, {"terminal": "no_valid_moves"}
        tray = action // 64
        rem = action % 64
        row = rem // 8
        col = rem % 8
        result = self.env.step((tray, row, col))
        obs = self._pack_obs(self.env._get_state())
        return obs, float(result.reward), bool(result.done), False, result.info

    def action_masks(self):
        mask = np.zeros(3 * 8 * 8, dtype=bool)
        for tray_index, block in enumerate(self.env.batch):
            if self.env.batch_used[tray_index]:
                continue
            block_h, block_w = block.shape
            for row in range(8 - block_h + 1):
                for col in range(8 - block_w + 1):
                    if valid_placement(self.env.board, block, row, col):
                        action_idx = tray_index * 64 + row * 8 + col
                        mask[action_idx] = True
        if not mask.any():
            if self._debug_mask:
                print("[mask] no valid actions; board:")
                for row in self.env.board:
                    print("".join("#" if v else "." for v in row))
                print("[mask] batch used:", self.env.batch_used)
                print("[mask] batch shapes:", [b.shape for b in self.env.batch])
            self._no_valid_moves = True
            mask[:] = False
            mask[0] = True
        return mask


def train_ppo(
    total_timesteps=10_000,
    model_path="ppo_blockblast.zip",
    save_freq=1_000,
    save_dir="ppo_checkpoints",
    use_masking=False,
    resume_step=None,
    use_sim=False,
    num_envs=1,
    log_episodes=1000,
    debug_masks=False,
    sim_model_batch=False,
):
    env = None
    wrapper_class = None
    wrapper_kwargs = None
    resume_path = None
    if resume_step is not None:
        resume_path = f"{save_dir}/rl_model_{resume_step}_steps.zip"
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_dir)
    if use_masking:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError(
                "Masking requires sb3-contrib. Install with: pip install sb3-contrib"
            ) from exc
    if use_sim:
        env = make_vec_env(
            BlockBlastSimGymEnv,
            n_envs=num_envs,
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs={"use_model_batch": sim_model_batch},
        )
    else:
        env = BlockBlastGymEnv()
        if wrapper_class is not None:
            env = wrapper_class(env, **wrapper_kwargs)
    if debug_masks:
        try:
            env.get_wrapper_attr("_debug_mask")
            env.set_attr("_debug_mask", True)
        except Exception:
            pass
    env = _wrap_monitor(env)
    policy_kwargs = {"net_arch": [256, 256, 128]}
    if use_masking:
        if resume_path:
            model = MaskablePPO.load(resume_path, env=env)
        else:
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                gamma=0.95,
                gae_lambda=0.95,
                n_steps=64,
                policy_kwargs=policy_kwargs,
            )
    else:
        if resume_path:
            model = PPO.load(resume_path, env=env)
        else:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                gamma=0.90,
                gae_lambda=0.85,
                n_steps=128,
                policy_kwargs=policy_kwargs,
            )
    callbacks = [checkpoint_callback]
    if log_episodes and log_episodes > 0:
        callbacks.append(EpisodeLogger(log_every=log_episodes))
    print(model.policy)
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
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
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--log-episodes", type=int, default=1000)
    parser.add_argument("--debug-masks", action="store_true")
    parser.add_argument("--sim-model-batch", action="store_true")
    args = parser.parse_args()
    train_ppo(
        total_timesteps=args.timesteps,
        model_path=args.model_path,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        use_masking=args.masking,
        resume_step=args.resume_step,
        use_sim=args.sim,
        num_envs=args.num_envs,
        log_episodes=args.log_episodes,
        debug_masks=args.debug_masks,
        sim_model_batch=args.sim_model_batch,
    )
