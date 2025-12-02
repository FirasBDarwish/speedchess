import os
import pickle
from dataclasses import dataclass
from typing import Dict, Any, List

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
from pydantic import BaseModel
from tqdm import trange
import wandb

from jax import debug as jdebug  # for optional jax.debug.print

from speed_gardner_chess import (
    GardnerChess,
    MAX_TERMINATION_STEPS,
    _step_board,   # pure board step for MCTS
    _observe,      # board-only observation
    DEFAULT_TIME,
)
from network_intermediate import AZNet # expose the intermediate


# ================================================================
# 1. Configs / hyperparams
# ================================================================

CKPT_ROOT = "/scratch/fbd2014/speedchess/examples/alphazero/nsims_checkpoints"
BASE_ENV_ID = "gardner_chess"      # non-speed env used for training base models
ITER_FILE = "000400.ckpt"          # which checkpoint to use for each nsim_*
# BASE_NSIMS = [8, 32]               # gate chooses between nsim=8 and nsim=32

SEED = 0

# PPO hyperparams
NUM_UPDATES = 1000                 # training iterations
ROLLOUT_STEPS = 2048               # gating decisions per update (across many eps)
GAMMA = 0.99
LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_CLIP_EPS = 0.2
PPO_LR = 3e-4
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.01
BATCH_SIZE = 256                   # minibatch size for PPO

# # Tick costs = number of MCTS simulations
# COST_8 = 8
# COST_32 = 32

# How often to log a sample game trajectory to wandb
TRAJ_LOG_INTERVAL = 50

# Gate checkpoint directory
GATE_CKPT_ROOT = "./gate_checkpoints"

# Which gating architecture to use:
# 1 = AZ intermediate + AZ value + time
# 2 = (AZ intermediate + raw obs) + AZ value + time
GATE_SETUP = 2  # or 2
NUM_ENVS = 64

assert ROLLOUT_STEPS % NUM_ENVS == 0, "ROLLOUT_STEPS must be divisible by NUM_ENVS"
STEPS_PER_ENV = ROLLOUT_STEPS // NUM_ENVS

# ================================================================
# Customizable gating configuration
# ================================================================
SIM_A = 8     # formerly COST_8
SIM_B = 32    # formerly COST_32

# Automatically use these as base nsims:
BASE_NSIMS = [SIM_A, SIM_B]

# wandb project name (prevents overwriting)
WANDB_PROJECT = f"pgx-speed-gate-{SIM_A}-vs-{SIM_B}"

# ================================================================
# 2. Stub Config + checkpoint loading (as in training)
# ================================================================

class TrainConfig(BaseModel):
    env_id: pgx.EnvId = "gardner_chess"
    seed: int = 0
    max_num_iters: int = 400
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    eval_interval: int = 5

    class Config:
        extra = "forbid"


class ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Config":
            return TrainConfig
        return super().find_class(module, name)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        data = ConfigUnpickler(f).load()
    model = data["model"]               # (params, state)
    cfg: TrainConfig = data["config"]
    env_id = data.get("env_id", cfg.env_id)
    return env_id, cfg, model


def discover_checkpoints(root: str, env_id: str, iter_filename: str) -> Dict[str, str]:
    """
    Find checkpoints like:
      root/nsim_8/gardner_chess_YYYY.../000400.ckpt
      root/nsim_32/gardner_chess_YYYY.../000400.ckpt
    Returns a dict { "nsim_8": path, "nsim_32": path, ... }.
    """
    ckpts: Dict[str, str] = {}
    if not os.path.isdir(root):
        raise ValueError(f"Checkpoint root '{root}' does not exist")

    for subdir in os.listdir(root):
        if not subdir.startswith("nsim_"):
            continue
        nsim = subdir.split("_", 1)[1]
        base = os.path.join(root, subdir)
        if not os.path.isdir(base):
            continue

        run_dirs = [d for d in os.listdir(base) if d.startswith(env_id)]
        if not run_dirs:
            continue
        run_dirs.sort()
        run_dir = os.path.join(base, run_dirs[-1])

        ckpt_path = os.path.join(run_dir, iter_filename)
        if os.path.exists(ckpt_path):
            ckpts[f"nsim_{nsim}"] = ckpt_path
    return ckpts


# ================================================================
# 3. Build AZ forward + MCTS selectors (for speed env)
# ================================================================

def build_forward(env, cfg: TrainConfig):
    def forward_fn(x, is_eval: bool = False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=cfg.num_channels,
            num_blocks=cfg.num_layers,
            resnet_v2=cfg.resnet_v2,
        )
        # AZNet now returns (policy_out, value_out, intermediate)
        policy_out, value_out, intermediate = net(
            x, is_training=not is_eval, test_local_stats=False
        )
        return policy_out, value_out, intermediate

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_recurrent_fn_speed(forward):
    """
    Recurrent fn for MCTS that only uses *board* dynamics (no clocks),
    so searches don't spend time.
    """

    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
        del rng_key
        model_params, model_state = model
        current_player = state.current_player

        # pure board step, batched
        state = jax.vmap(_step_board)(state, action)

        # update observation for side-to-move (board-only features)
        obs = jax.vmap(_observe)(state, state.current_player)
        state = state.replace(observation=obs)

        (logits, value, _intermediate), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        batch_size = state.rewards.shape[0]
        reward = state.rewards[jnp.arange(batch_size), current_player]

        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), state

    return recurrent_fn


def make_select_actions_mcts(forward, recurrent_fn, num_simulations: int):
    def select_actions_mcts(model, state, rng_key):
        model_params, model_state = model
        (logits, value, _intermediate), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=int(num_simulations),
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action  # (batch,)

    return select_actions_mcts


# ================================================================
# 4. GateNetV2:
# ================================================================

class GateNetV2(hk.Module):
    """
    Gating network that can operate in two modes:

    mode=1:
      - process AZNet intermediate feature map with a small conv trunk
      - concat time_left_norm + AZ value
      - MLP → logits (2-way) + scalar value

    mode=2:
      - process AZNet intermediate feature map with conv trunk
      - process raw observation with conv trunk (like original GateNet)
      - concat both + time_left_norm + AZ value
      - MLP → logits (2-way) + scalar value
    """

    def __init__(self, num_options: int = 2, mode: int = 1, name: str = "GateNetV2"):
        super().__init__(name=name)
        self.num_options = num_options
        assert mode in (1, 2)
        self.mode = mode

    def __call__(self, obs, time_left_norm, az_value, az_intermediate):
        """
        obs:            (B, 5, 5, 115)
        time_left_norm: (B, 2)   normalized [my_time, opp_time]
        az_value:       (B,)     AZNet scalar value prediction
        az_intermediate:(B, 5, 5, C) feature map from AZNet trunk
        """
        # ---------- Branch on AZ intermediate ----------
        x_az = az_intermediate.astype(jnp.float32)   # (B,5,5,C)
        conv_az = hk.Sequential([
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Flatten(),
        ])
        z_az = conv_az(x_az)  # (B, hidden_az)

        # ---------- Optional branch on raw observation ----------
        if self.mode == 2:
            x_obs = obs.astype(jnp.float32)          # (B,5,5,115)
            # keep the same pattern as your original GateNet
            x_obs = jnp.moveaxis(x_obs, -1, 1)       # (B,115,5,5)

            conv_obs = hk.Sequential([
                hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
                jax.nn.relu,
                hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
                jax.nn.relu,
                hk.Flatten(),
            ])
            z_obs = conv_obs(x_obs)                  # (B, hidden_obs)
            z = jnp.concatenate([z_az, z_obs], axis=-1)
        else:
            # mode 1: only AZ intermediate branch
            z = z_az

        # ---------- Add time + AZ value ----------
        time_left_norm = time_left_norm.astype(jnp.float32)  # (B,2)
        az_value = az_value.astype(jnp.float32)              # (B,)
        z = jnp.concatenate([z, time_left_norm, az_value[..., None]], axis=-1)

        # ---------- Shared MLP → logits + value ----------
        h = hk.Linear(128)(z)
        h = jax.nn.relu(h)

        logits = hk.Linear(self.num_options)(h)              # (B,2)
        value = hk.Linear(1)(h)[..., 0]                      # (B,)

        return logits, value


def gate_forward_fn(obs_batch, time_batch, az_value_batch, az_inter_batch):
    """
    Wrapper that instantiates GateNetV2 with the chosen mode.
    """
    net = GateNetV2(num_options=2, mode=GATE_SETUP)
    return net(obs_batch, time_batch, az_value_batch, az_inter_batch)


gate_forward = hk.without_apply_rng(hk.transform(gate_forward_fn))


# ================================================================
# 5. PPO loss, GAE helper
# ================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class PPOBatch:
    obs: jnp.ndarray         # (T or U, B, 5,5,115) or (T,5,5,115)
    time: jnp.ndarray        # (T or U, B, 2) or (T,2)
    actions: jnp.ndarray     # (T or U, B) or (T,)
    logp_old: jnp.ndarray    # (T or U, B) or (T,)
    values_old: jnp.ndarray  # (T or U, B) or (T,)
    returns: jnp.ndarray     # (T or U, B) or (T,)
    advantages: jnp.ndarray  # (T or U, B) or (T,)

    # AZNet features used by the gate
    az_value: jnp.ndarray    # (T or U, B) or (T,)
    az_inter: jnp.ndarray    # (T or U, B, 5,5,C) or (T,5,5,C)

    def tree_flatten(self):
        children = (
            self.obs,
            self.time,
            self.actions,
            self.logp_old,
            self.values_old,
            self.returns,
            self.advantages,
            self.az_value,
            self.az_inter,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            obs,
            time,
            actions,
            logp_old,
            values_old,
            returns,
            advantages,
            az_value,
            az_inter,
        ) = children
        return cls(
            obs=obs,
            time=time,
            actions=actions,
            logp_old=logp_old,
            values_old=values_old,
            returns=returns,
            advantages=advantages,
            az_value=az_value,
            az_inter=az_inter,
        )


@jax.jit
def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones: (T, B)
    Returns: returns, advantages: (T, B)
    """
    rev_rewards = rewards[::-1]
    rev_values = values[::-1]
    rev_dones = dones[::-1]

    def body_fn(carry, inp):
        next_value, next_adv = carry        # (B,), (B,)
        r, v, d = inp                       # (B,), (B,), (B,)
        mask = 1.0 - d
        delta = r + gamma * next_value * mask - v
        adv = delta + gamma * lam * mask * next_adv
        ret = adv + v
        return (v, adv), (ret, adv)

    init_value = jnp.zeros_like(values[0])  # (B,)
    init_adv = jnp.zeros_like(values[0])    # (B,)

    (_, _), (rev_returns, rev_advantages) = jax.lax.scan(
        body_fn,
        (init_value, init_adv),
        (rev_rewards, rev_values, rev_dones),
    )

    returns = rev_returns[::-1]
    advantages = rev_advantages[::-1]
    return returns, advantages

def make_ppo_loss_fn():
    def loss_fn(params, batch: PPOBatch):
        logits, values = gate_forward.apply(
            params,
            batch.obs,
            batch.time,
            batch.az_value,
            batch.az_inter,
        )  # logits: (...,2), values: (...)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        logp = jnp.take_along_axis(
            log_probs,
            batch.actions[..., None],
            axis=-1,
        )[..., 0]

        # Ratio
        ratios = jnp.exp(logp - batch.logp_old)

        # Normalize advantages
        adv_mean = jnp.mean(batch.advantages)
        adv_std = jnp.std(batch.advantages) + 1e-8
        adv_norm = (batch.advantages - adv_mean) / adv_std

        # Clipped surrogate
        unclipped = ratios * adv_norm
        clipped = jnp.clip(ratios, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_norm
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        # Value loss
        value_loss = jnp.mean((batch.returns - values) ** 2)

        # Entropy
        probs = jnp.exp(log_probs)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

        loss = (
            policy_loss
            + PPO_VF_COEF * value_loss
            - PPO_ENT_COEF * entropy
        )

        approx_kl = 0.5 * jnp.mean((logp - batch.logp_old) ** 2)
        metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }
        return loss, metrics

    return loss_fn


# ================================================================
# 6. Helper: build PPOBatch + rollout stats + histograms from traj
# ================================================================

def build_batch_and_stats(traj: Dict[str, jnp.ndarray]):
    """
    traj keys (all JAX arrays with shape (T,B,...)):
      obs:      (T,B,5,5,115)
      time:     (T,B,2)
      action:   (T,B)
      logp:     (T,B)
      value:    (T,B)
      reward:   (T,B)
      done:     (T,B)
      player:   (T,B)
      az_value: (T,B)
      az_inter: (T,B,5,5,C)
    """
    obs_arr = traj["obs"]
    time_arr = traj["time"]
    actions_arr = traj["action"]
    logp_arr = traj["logp"]
    values_arr = traj["value"]
    rewards_arr = traj["reward"]
    dones_arr = traj["done"]
    players_arr = traj["player"]
    az_value_arr = traj["az_value"]
    az_inter_arr = traj["az_inter"]

    T, B = actions_arr.shape

    # ---------- GAE on device ----------
    returns, advantages = compute_gae(
        rewards_arr,
        values_arr,
        dones_arr.astype(jnp.float32),
        GAMMA,
        LAMBDA,
    )  # (T,B)

    # ---------- Flatten (T,B) -> (T*,) for PPO ----------
    T_star = T * B

    obs_flat        = obs_arr.reshape(T_star, 5, 5, 115)
    time_flat       = time_arr.reshape(T_star, 2)
    actions_flat    = actions_arr.reshape(T_star)
    logp_flat       = logp_arr.reshape(T_star)
    values_flat     = values_arr.reshape(T_star)
    returns_flat    = returns.reshape(T_star)
    advantages_flat = advantages.reshape(T_star)
    az_value_flat   = az_value_arr.reshape(T_star)
    # keep last dims (5,5,C) intact
    az_inter_flat   = az_inter_arr.reshape(T_star, *az_inter_arr.shape[2:])

    batch = PPOBatch(
        obs=obs_flat,
        time=time_flat,
        actions=actions_flat,
        logp_old=logp_flat,
        values_old=values_flat,
        returns=returns_flat,
        advantages=advantages_flat,
        az_value=az_value_flat,
        az_inter=az_inter_flat,
    )

    # ---------- Stats on host ----------
    actions_np  = np.array(actions_arr, dtype=np.int32)        # (T,B)
    players_np  = np.array(players_arr, dtype=np.int32)        # (T,B)
    dones_np    = np.array(dones_arr > 0.5, dtype=bool)        # (T,B)
    rewards_np  = np.array(rewards_arr, dtype=np.float32)      # (T,B)
    adv_np      = np.array(advantages, dtype=np.float32)       # (T,B)
    ret_np      = np.array(returns, dtype=np.float32)          # (T,B)
    time_np     = np.array(time_arr, dtype=np.float32)         # (T,B,2)

    # Episode lengths & returns aggregated over all envs
    ep_lens_list: List[int] = []
    ep_returns_list: List[float] = []

    for b in range(B):
        done_idx = np.where(dones_np[:, b])[0]
        if len(done_idx) == 0:
            continue
        start = 0
        for idx in done_idx:
            length = idx - start + 1
            ep_lens_list.append(length)
            ep_returns_list.append(float(rewards_np[start:idx + 1, b].sum()))
            start = idx + 1

    if len(ep_lens_list) == 0:
        ep_lens = np.array([T], dtype=np.int32)
        ep_returns = np.array([rewards_np.sum()], dtype=np.float32)
    else:
        ep_lens = np.array(ep_lens_list, dtype=np.int32)
        ep_returns = np.array(ep_returns_list, dtype=np.float32)

    stats: Dict[str, float] = {
        "mean_ep_return": float(ep_returns.mean()),
        "mean_ep_len": float(ep_lens.mean()),
    }

    # ---------- Action usage stats ----------
    actions_flat_np  = actions_np.reshape(-1)
    players_flat_np  = players_np.reshape(-1)
    total_steps      = max(1, len(actions_flat_np))

    rate_8  = float((actions_flat_np == 0).sum() / total_steps)
    rate_32 = float((actions_flat_np == 1).sum() / total_steps)

    mask0   = players_flat_np == 0
    mask1   = players_flat_np == 1
    total_p0 = max(1, int(mask0.sum()))
    total_p1 = max(1, int(mask1.sum()))

    rate_8_p0 = float((actions_flat_np[mask0] == 0).sum() / total_p0) if total_p0 > 0 else 0.0
    rate_8_p1 = float((actions_flat_np[mask1] == 0).sum() / total_p1) if total_p1 > 0 else 0.0

    stats.update(
        {
            "action_rate_8": rate_8,
            "action_rate_32": rate_32,
            "action_rate_8_p0": rate_8_p0,
            "action_rate_8_p1": rate_8_p1,
        }
    )

    # ---------- Move counts ----------
    num_games = max(1, len(ep_lens))
    moves_p0 = int((players_flat_np == 0).sum())
    moves_p1 = int((players_flat_np == 1).sum())
    avg_moves_per_game = float(ep_lens.mean())
    stats.update(
        {
            "avg_moves_per_game": avg_moves_per_game,
            "avg_moves_p0": float(moves_p0) / num_games,
            "avg_moves_p1": float(moves_p1) / num_games,
        }
    )

    # ---------- Advantage / return stats ----------
    stats.update(
        {
            "advantages_mean": float(adv_np.mean()),
            "advantages_std": float(adv_np.std()),
            "returns_mean": float(ret_np.mean()),
            "returns_std": float(ret_np.std()),
        }
    )

    # ---------- W/L/D from player-0 perspective ----------
    wins_p0 = 0
    losses_p0 = 0
    draws = 0

    hist_vals_8: List[int] = []
    hist_vals_32: List[int] = []
    ep_traces: List[dict] = []

    for b in range(B):
        done_idx = np.where(dones_np[:, b])[0]
        if len(done_idx) == 0:
            continue
        start = 0
        for idx in done_idx:
            steps: List[dict] = []
            local_move_idx = 0
            for t in range(start, idx + 1):
                a = int(actions_np[t, b])
                if a == 0:
                    hist_vals_8.append(local_move_idx)
                else:
                    hist_vals_32.append(local_move_idx)

                steps.append(
                    {
                        "player": int(players_np[t, b]),
                        "action": a,
                        "my_time": float(time_np[t, b, 0]),
                        "opp_time": float(time_np[t, b, 1]),
                    }
                )
                local_move_idx += 1

            # Final outcome from rewards: reward is for prev_player at termination
            r = float(rewards_np[idx, b])
            mover = int(players_np[idx, b])
            if r > 0:
                if mover == 0:
                    result_p0 = r
                    result_p1 = -r
                else:
                    result_p1 = r
                    result_p0 = -r
            elif r < 0:
                if mover == 0:
                    result_p0 = r
                    result_p1 = -r
                else:
                    result_p1 = r
                    result_p0 = -r
            else:
                result_p0 = 0.0
                result_p1 = 0.0

            if result_p0 > 0:
                wins_p0 += 1
            elif result_p0 < 0:
                losses_p0 += 1
            else:
                draws += 1

            # only keep a few example traces
            if len(ep_traces) < 5:
                ep_traces.append(
                    {
                        "steps": steps,
                        "result_p0": float(result_p0),
                        "result_p1": float(result_p1),
                    }
                )

            start = idx + 1

    num_games_wld = max(1, wins_p0 + losses_p0 + draws)
    stats.update(
        {
            "win_rate_p0": wins_p0 / num_games_wld,
            "loss_rate_p0": losses_p0 / num_games_wld,
            "draw_rate": draws / num_games_wld,
        }
    )

    hist_vals_8 = np.array(hist_vals_8, dtype=np.int32)
    hist_vals_32 = np.array(hist_vals_32, dtype=np.int32)

    return batch, stats, ep_traces, hist_vals_8, hist_vals_32

# ================================================================
# 7. Jitted rollout (scan over time) – gate + random gate
# ================================================================

def make_rollout_core(
    env_speed: GardnerChess,
    forward,            # AZNet forward (with intermediate)
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
):
    """
    Batched self-play rollout over NUM_ENVS envs.
    state: batched State, leading dim = NUM_ENVS
    """
    default_time_f32 = jnp.float32(default_time)
    feat_params, feat_state = model_32   # use nsim_32 model to extract AZ features

    def rollout_core(state, gate_params, rng_key, num_steps: int):
        num_envs = NUM_ENVS

        def env_reset(key):
            return env_speed.init(key)

        def step_fn(carry, _):
            state, rng = carry

            # Split RNG
            rng, key_reset, key_gate, key_mcts8, key_mcts32 = jax.random.split(rng, 5)

            # Reset envs that were terminal/truncated
            done_prev = state.terminated | state.truncated   # (B,)
            keys_reset = jax.random.split(key_reset, num_envs)

            def reset_one(s, k, d):
                return jax.lax.cond(
                    d,
                    lambda _: env_reset(k),
                    lambda _: s,
                    operand=None,
                )

            state = jax.vmap(reset_one)(state, keys_reset, done_prev)

            # Side-to-move observation (batched)
            obs = state.observation               # (B,5,5,115)
            time_left = state.time_left           # (B,2)
            cur = state.current_player            # (B,)

            cur_idx = cur[:, None]
            opp_idx = (1 - cur)[:, None]
            my_time = jnp.take_along_axis(time_left, cur_idx, axis=1)[:, 0]
            opp_time = jnp.take_along_axis(time_left, opp_idx, axis=1)[:, 0]

            time_norm = jnp.stack(
                [my_time / default_time_f32, opp_time / default_time_f32],
                axis=1,
            )  # (B,2)

            # -------- AZNet features (batched) --------
            (az_logits_b, az_value_b, az_inter_b), _ = forward.apply(
                feat_params, feat_state, obs, is_eval=True
            )
            # az_value_b: (B,), az_inter_b: (B,5,5,C)

            # -------- Gate decision (GateNetV2, batched) --------
            logits, value = gate_forward.apply(
                gate_params, obs, time_norm, az_value_b, az_inter_b
            )  # logits: (B,2), value: (B,)

            log_probs = jax.nn.log_softmax(logits, axis=-1)
            gate_action = jax.random.categorical(key_gate, logits, axis=-1)  # (B,)
            batch_idx = jnp.arange(num_envs)
            logp = log_probs[batch_idx, gate_action]                         # (B,)

            # -------- Execute move via chosen MCTS per env --------
            key_mcts_env8 = jax.random.split(key_mcts8, num_envs)
            key_mcts_env32 = jax.random.split(key_mcts32, num_envs)

            def run_mcts_for_env(s, g, k8, k32):
                s_b = jax.tree_util.tree_map(lambda x: x[None, ...], s)

                def run8(_):
                    a = select_mcts_8(model_8, s_b, k8)[0]
                    return a, jnp.int32(SIM_A)

                def run32(_):
                    a = select_mcts_32(model_32, s_b, k32)[0]
                    return a, jnp.int32(SIM_B)

                return jax.lax.cond(
                    g == 0,
                    run8,
                    run32,
                    operand=None,
                )

            actions, time_spent = jax.vmap(
                run_mcts_for_env,
                in_axes=(0, 0, 0, 0),
                out_axes=(0, 0),
            )(state, gate_action, key_mcts_env8, key_mcts_env32)

            prev_player = state.current_player     # (B,)

            def step_one(env_state, a, t):
                return env_speed.step(env_state, (a, t))

            state_next = jax.vmap(step_one, in_axes=(0, 0, 0))(
                state, actions, time_spent
            )

            done = (state_next.terminated | state_next.truncated).astype(jnp.float32)  # (B,)

            rewards_all = state_next.rewards                           # (B,2)
            prev_idx = prev_player[:, None]
            reward_prev = jnp.take_along_axis(rewards_all, prev_idx, axis=1)[:, 0]
            reward = jnp.where(done > 0.0, reward_prev, jnp.float32(0.0))

            carry_next = (state_next, rng)
            out = {
                "obs": obs,
                "time": time_norm,
                "action": gate_action,
                "logp": logp,
                "value": value,
                "reward": reward,
                "done": done,
                "player": prev_player,
                "az_value": az_value_b,
                "az_inter": az_inter_b,
            }
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            step_fn, (state, rng_key), xs=None, length=num_steps
        )
        return state_final, rng_final, traj

    rollout_core_jit = jax.jit(rollout_core, static_argnums=(3,))
    return rollout_core_jit

def make_rollout_core_random(
    env_speed: GardnerChess,
    forward,
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
):
    """
    Random gating baseline, batched over NUM_ENVS envs.
    """
    default_time_f32 = jnp.float32(default_time)
    feat_params, feat_state = model_32

    def rollout_core_random(state, gate_params_unused, rng_key, num_steps: int):
        num_envs = NUM_ENVS

        def env_reset(key):
            return env_speed.init(key)

        def step_fn(carry, _):
            state, rng = carry

            rng, key_reset, key_gate, key_mcts8, key_mcts32 = jax.random.split(rng, 5)

            # Reset envs done in previous step
            done_prev = state.terminated | state.truncated   # (B,)
            keys_reset = jax.random.split(key_reset, num_envs)

            def reset_one(s, k, d):
                return jax.lax.cond(
                    d,
                    lambda _: env_reset(k),
                    lambda _: s,
                    operand=None,
                )

            state = jax.vmap(reset_one)(state, keys_reset, done_prev)

            # Side-to-move observation
            obs = state.observation               # (B,5,5,115)
            time_left = state.time_left           # (B,2)
            cur = state.current_player            # (B,)

            cur_idx = cur[:, None]
            opp_idx = (1 - cur)[:, None]
            my_time = jnp.take_along_axis(time_left, cur_idx, axis=1)[:, 0]
            opp_time = jnp.take_along_axis(time_left, opp_idx, axis=1)[:, 0]

            time_norm = jnp.stack(
                [my_time / default_time_f32, opp_time / default_time_f32],
                axis=1,
            )  # (B,2)

            # AZ features (not used by random gate, but for logging consistency)
            (az_logits_b, az_value_b, az_inter_b), _ = forward.apply(
                feat_params, feat_state, obs, is_eval=True
            )

            # Random gate: uniform {0,1}
            gate_action = jax.random.randint(key_gate, (num_envs,), 0, 2)  # (B,)
            logp = jnp.log(jnp.full((num_envs,), 0.5, dtype=jnp.float32))
            value = jnp.zeros((num_envs,), dtype=jnp.float32)

            # Execute selected MCTS per env
            key_mcts_env8 = jax.random.split(key_mcts8, num_envs)
            key_mcts_env32 = jax.random.split(key_mcts32, num_envs)

            def run_mcts_for_env(s, g, k8, k32):
                s_b = jax.tree_util.tree_map(lambda x: x[None, ...], s)

                def run8(_):
                    a = select_mcts_8(model_8, s_b, k8)[0]
                    return a, jnp.int32(SIM_A)

                def run32(_):
                    a = select_mcts_32(model_32, s_b, k32)[0]
                    return a, jnp.int32(SIM_B)

                return jax.lax.cond(
                    g == 0,
                    run8,
                    run32,
                    operand=None,
                )

            actions, time_spent = jax.vmap(
                run_mcts_for_env,
                in_axes=(0, 0, 0, 0),
                out_axes=(0, 0),
            )(state, gate_action, key_mcts_env8, key_mcts_env32)

            prev_player = state.current_player     # (B,)

            def step_one(env_state, a, t):
                return env_speed.step(env_state, (a, t))

            state_next = jax.vmap(step_one, in_axes=(0, 0, 0))(
                state, actions, time_spent
            )

            done = (state_next.terminated | state_next.truncated).astype(jnp.float32)  # (B,)

            rewards_all = state_next.rewards                           # (B,2)
            prev_idx = prev_player[:, None]
            reward_prev = jnp.take_along_axis(rewards_all, prev_idx, axis=1)[:, 0]
            reward = jnp.where(done > 0.0, reward_prev, jnp.float32(0.0))

            carry_next = (state_next, rng)
            out = {
                "obs": obs,
                "time": time_norm,
                "action": gate_action,
                "logp": logp,
                "value": value,
                "reward": reward,
                "done": done,
                "player": prev_player,
                "az_value": az_value_b,
                "az_inter": az_inter_b,
            }
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            step_fn, (state, rng_key), xs=None, length=num_steps
        )
        return state_final, rng_final, traj

    rollout_core_random_jit = jax.jit(rollout_core_random, static_argnums=(3,))
    return rollout_core_random_jit

def rollout_selfplay(
    rollout_core_jit,
    state,
    gate_params,
    rng_key,
    num_steps: int,
):
    """
    Wrapper: calls jitted rollout_core, then builds PPOBatch + stats + episode traces + hist.
    """
    state_final, rng_final, traj = rollout_core_jit(
        state, gate_params, rng_key, num_steps
    )

    batch, stats, ep_traces, hist_vals_8, hist_vals_32 = build_batch_and_stats(traj)
    return state_final, rng_final, batch, stats, ep_traces, hist_vals_8, hist_vals_32

def rollout_selfplay_batched(
    rollout_core_jit,
    states,         # batched State, leading dim = NUM_ENVS
    gate_params,
    rng_key,
    num_steps: int,
    num_envs: int,
):
    """
    Run self-play in NUM_ENVS parallel environments using vmap around
    the single-env rollout_core_jit.

    Returns:
      states_final: batched State
      rng_out:      new RNG key (for future calls)
      batch:        PPOBatch with shape (T*num_envs, ...)
      stats, ep_traces, hist_vals_8, hist_vals_32: as before
    """

    # Split a key for each env
    rng_key, rollout_key = jax.random.split(rng_key)
    env_keys = jax.random.split(rollout_key, num_envs)

    def one_env_rollout(state, key):
        # rollout_core_jit already returns (state_final, rng_final, traj)
        state_f, rng_f, traj = rollout_core_jit(
            state, gate_params, key, num_steps
        )
        # We don't actually *need* rng_f if we always resplit from rng_key outside
        return state_f, traj

    # vmap over the env dimension
    states_final, traj = jax.vmap(
        one_env_rollout,
        in_axes=(0, 0),
        out_axes=(0, 0),
    )(states, env_keys)

    # traj is a dict of arrays with shape (NUM_ENVS, T, ...)
    # Flatten env + time into a single time-like dimension: (NUM_ENVS*T, ...)
    traj_flat = {
        k: v.reshape(-1, *v.shape[2:]) for k, v in traj.items()
    }

    batch, stats, ep_traces, hist_vals_8, hist_vals_32 = build_batch_and_stats(
        traj_flat
    )

    # rng_key already advanced via split above
    rng_out = rng_key
    return states_final, rng_out, batch, stats, ep_traces, hist_vals_8, hist_vals_32


# ================================================================
# 8. Main training loop (self-play PPO) + wandb + lax.scan over minibatches
# ================================================================

def main():
    # ----------------------
    # Load base models
    # ----------------------
    ckpt_paths = discover_checkpoints(CKPT_ROOT, BASE_ENV_ID, ITER_FILE)
    print("found checkpoints: ", ckpt_paths)

    needed = [f"nsim_{n}" for n in BASE_NSIMS]
    for k in needed:
        if k not in ckpt_paths:
            raise RuntimeError(f"Missing checkpoint for {k} in {CKPT_ROOT}")

    env_id_8, cfg_8, model_8 = load_checkpoint(ckpt_paths[f"nsim_{SIM_A}"])
    env_id_32, cfg_32, model_32 = load_checkpoint(ckpt_paths[f"nsim_{SIM_B}"])
    print("loaded checkpoints: ", env_id_8, env_id_32)

    if env_id_8 != BASE_ENV_ID or env_id_32 != BASE_ENV_ID:
        raise RuntimeError("Base model env_ids mismatch")

    # Check architecture consistency
    if (
        cfg_8.num_layers != cfg_32.num_layers
        or cfg_8.num_channels != cfg_32.num_channels
        or cfg_8.resnet_v2 != cfg_32.resnet_v2
    ):
        raise RuntimeError("Base models have different architectures!")

    # ----------------------
    # Speed env + forward net
    # ----------------------
    env_speed = GardnerChess()
    rng = jax.random.PRNGKey(SEED)
    rng, key_init = jax.random.split(rng)
    
    init_keys = jax.random.split(key_init, NUM_ENVS)
    state = jax.vmap(env_speed.init)(init_keys)  # batched State, leading dim = NUM_ENVS

    # Use initial clock as "default_time" for normalization
    default_time = float(DEFAULT_TIME)

    forward = build_forward(env_speed, cfg_8)  # same arch for both base models

    # ----------------------
    # MCTS selectors (shared nets for both players)
    # ----------------------
    recurrent_fn_speed = make_recurrent_fn_speed(forward)
    select_mcts_8 = make_select_actions_mcts(forward, recurrent_fn_speed, SIM_A)
    select_mcts_32 = make_select_actions_mcts(forward, recurrent_fn_speed, SIM_B)

    # ----------------------
    # Init GateNet + optimizer
    # ----------------------
    dummy_obs = jnp.zeros((1, 5, 5, 115), dtype=jnp.float32)
    dummy_time = jnp.zeros((1, 2), dtype=jnp.float32)

    # AZNet value is scalar per position, so shape (B,)
    dummy_az_value = jnp.zeros((1,), dtype=jnp.float32)

    # AZNet intermediate has shape (B,5,5,num_channels)
    dummy_az_inter = jnp.zeros((1, 5, 5, cfg_8.num_channels), dtype=jnp.float32)

    gate_params = gate_forward.init(
        jax.random.PRNGKey(SEED + 123),
        dummy_obs,
        dummy_time,
        dummy_az_value,
        dummy_az_inter,
    )


    optimizer = optax.adam(PPO_LR)
    opt_state = optimizer.init(gate_params)

    loss_fn = make_ppo_loss_fn()
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def ppo_update_many(params, opt_state, batches: PPOBatch):
        """
        batches: PPOBatch where each field has shape (U, B, ...)
                 U = total number of SGD steps (epochs * minibatches)
        """

        def body(carry, mb: PPOBatch):
            params, opt_state = carry
            (loss, metrics), grads = value_and_grad(params, mb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), metrics

        (params_final, opt_state_final), metrics_seq = jax.lax.scan(
            body,
            (params, opt_state),
            batches,       # PyTree with leading dim U
        )
        # metrics_seq is a dict of arrays, each (U,)
        return params_final, opt_state_final, metrics_seq

    # ----------------------
    # Jitted rollout cores (gate + random)
    # ----------------------
    rollout_core_jit = make_rollout_core(
        env_speed,
        forward,
        select_mcts_8,
        select_mcts_32,
        model_8,
        model_32,
        default_time,
    )
    rollout_core_random_jit = make_rollout_core_random(
        env_speed,
        forward,
        select_mcts_8,
        select_mcts_32,
        model_8,
        model_32,
        default_time,
    )

    # ----------------------
    # wandb init
    # ----------------------
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "ckpt_root": CKPT_ROOT,
            "base_env_id": BASE_ENV_ID,
            "iter_file": ITER_FILE,
            "base_nsims": BASE_NSIMS,
            "seed": SEED,
            "num_updates": NUM_UPDATES,
            "rollout_steps": ROLLOUT_STEPS,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "ppo_epochs": PPO_EPOCHS,
            "ppo_clip_eps": PPO_CLIP_EPS,
            "ppo_lr": PPO_LR,
            "ppo_vf_coef": PPO_VF_COEF,
            "ppo_ent_coef": PPO_ENT_COEF,
            "batch_size": BATCH_SIZE,
            "sim_a": SIM_A,
            "sim_b": SIM_B,
            "gate_setup": GATE_SETUP,
        },
        name=WANDB_PROJECT,
    )
    print("initialized wandb and checkpoint loading")

    run = wandb.run
    run_name = run.name if run is not None else "gate_run"
    gate_setup = int(os.getenv("GATE_SETUP", "1"))
    save_dir = os.path.join(
        GATE_CKPT_ROOT, f"gate_{SIM_A}vs{SIM_B}_setup{gate_setup}_{run_name}"
    )

    os.makedirs(save_dir, exist_ok=True)
    SAVE_INTERVAL = max(1, NUM_UPDATES // 10)
    best_mean_return = -1e9

    # ----------------------
    # Training loop
    # ----------------------
    for update in trange(1, NUM_UPDATES + 1, desc="PPO updates"):
        # Collect batched rollout (jitted + vmapped)
        state, rng, batch, roll_stats, ep_traces, hist_vals_8, hist_vals_32 = rollout_selfplay(
            rollout_core_jit,
            state,
            gate_params,
            rng,
            STEPS_PER_ENV,
        )

        # Shuffle batch and do minibatch PPO via single scan
        T = batch.obs.shape[0]
        num_mbs = T // BATCH_SIZE
        if num_mbs == 0:
            raise RuntimeError(f"Batch too small for BATCH_SIZE={BATCH_SIZE}, T={T}")
        
        # Total number of SGD minibatches = epochs * minibatches-per-epoch
        U = PPO_EPOCHS * num_mbs
        
        # Use JAX RNG instead of NumPy RNG for shuffling
        rng, key_perm = jax.random.split(rng)              # advance global RNG
        keys_epochs = jax.random.split(key_perm, PPO_EPOCHS)  # one key per epoch
        
        def make_epoch_idxs(k):
            """
            For a single epoch:
              - create a random permutation of [0, 1, ..., T-1]
              - keep exactly num_mbs * BATCH_SIZE indices
              - reshape into (num_mbs, BATCH_SIZE)
            """
            idxs = jax.random.permutation(k, T)            # shape (T,)
            idxs = idxs[: num_mbs * BATCH_SIZE]            # (num_mbs*B,)
            return idxs.reshape(num_mbs, BATCH_SIZE)       # (num_mbs, B)
        
        # Vectorise the above across epochs → (PPO_EPOCHS, num_mbs, BATCH_SIZE)
        all_mb_indices = jax.vmap(make_epoch_idxs)(keys_epochs)  # (E, num_mbs, B)
        
        # Flatten epochs × minibatches into one dimension U
        all_mb_indices = all_mb_indices.reshape(U, BATCH_SIZE)   # (U, B)
        
        # Now slice the big batch into a sequence of minibatches, just like before
        mb_obs        = batch.obs[all_mb_indices]         # (U,B,...)
        mb_time       = batch.time[all_mb_indices]
        mb_actions    = batch.actions[all_mb_indices]
        mb_logp       = batch.logp_old[all_mb_indices]
        mb_values     = batch.values_old[all_mb_indices]
        mb_returns    = batch.returns[all_mb_indices]
        mb_advantages = batch.advantages[all_mb_indices]
        mb_az_value   = batch.az_value[all_mb_indices]
        mb_az_inter   = batch.az_inter[all_mb_indices]
        
        batches_seq = PPOBatch(
            obs=mb_obs,
            time=mb_time,
            actions=mb_actions,
            logp_old=mb_logp,
            values_old=mb_values,
            returns=mb_returns,
            advantages=mb_advantages,
            az_value=mb_az_value,
            az_inter=mb_az_inter,
        )

        gate_params, opt_state, metrics_seq = ppo_update_many(
            gate_params, opt_state, batches_seq
        )

        metrics_avg = {k: float(jnp.mean(v)) for k, v in metrics_seq.items()}

        # --------------------------
        # Optional: random baseline & checkpointing every 10%
        # --------------------------
        random_stats = None
        if update % SAVE_INTERVAL == 0:
            # Save current gate
            ckpt_path = os.path.join(save_dir, f"gate_{update:06d}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "update": update,
                        "gate_params": gate_params,
                        "opt_state": opt_state,
                        "config": dict(wandb.config),
                        "rollout_stats": roll_stats,
                    },
                    f,
                )

            # Update best gate based on mean_ep_return
            if roll_stats["mean_ep_return"] > best_mean_return:
                best_mean_return = roll_stats["mean_ep_return"]
                best_ckpt_path = os.path.join(save_dir, "gate_best.pkl")
                with open(best_ckpt_path, "wb") as f:
                    pickle.dump(
                        {
                            "best_update": update,
                            "gate_params": gate_params,
                            "opt_state": opt_state,
                            "config": dict(wandb.config),
                            "rollout_stats": roll_stats,
                        },
                        f,
                    )

            # Random gating baseline evaluation
            rng, key_eval_init = jax.random.split(rng)
            eval_keys = jax.random.split(key_eval_init, NUM_ENVS)
            state_eval = jax.vmap(env_speed.init)(eval_keys)
            
            state_eval, rng, traj_random = rollout_core_random_jit(
                state_eval, gate_params, rng, STEPS_PER_ENV
            )
            _batch_rand, random_stats, _ep_traces_rand, _hist8_rand, _hist32_rand = build_batch_and_stats(traj_random)

        # Build wandb log dict
        log_dict: Dict[str, Any] = {
            "train/loss": metrics_avg["loss"],
            "train/policy_loss": metrics_avg["policy_loss"],
            "train/value_loss": metrics_avg["value_loss"],
            "train/entropy": metrics_avg["entropy"],
            "train/approx_kl": metrics_avg["approx_kl"],
            "rollout/mean_ep_return": roll_stats["mean_ep_return"],
            "rollout/mean_ep_len": roll_stats["mean_ep_len"],
            "rollout/avg_moves_per_game": roll_stats["avg_moves_per_game"],
            "rollout/avg_moves_p0": roll_stats["avg_moves_p0"],
            "rollout/avg_moves_p1": roll_stats["avg_moves_p1"],
            "rollout/action_rate_8": roll_stats["action_rate_8"],
            "rollout/action_rate_32": roll_stats["action_rate_32"],
            "rollout/action_rate_8_p0": roll_stats["action_rate_8_p0"],
            "rollout/action_rate_8_p1": roll_stats["action_rate_8_p1"],
            "rollout/advantages_mean": roll_stats["advantages_mean"],
            "rollout/advantages_std": roll_stats["advantages_std"],
            "rollout/returns_mean": roll_stats["returns_mean"],
            "rollout/returns_std": roll_stats["returns_std"],
            "rollout/win_rate_p0": roll_stats["win_rate_p0"],
            "rollout/loss_rate_p0": roll_stats["loss_rate_p0"],
            "rollout/draw_rate": roll_stats["draw_rate"],
        }

        # Histograms: move index where nsim=8 / nsim=32 were used
        if hist_vals_8.size > 0:
            log_dict["rollout/hist_move_idx_8"] = wandb.Histogram(hist_vals_8)
        if hist_vals_32.size > 0:
            log_dict["rollout/hist_move_idx_32"] = wandb.Histogram(hist_vals_32)

        # Random baseline logs (when computed)
        if random_stats is not None:
            log_dict.update(
                {
                    "random/mean_ep_return": random_stats["mean_ep_return"],
                    "random/mean_ep_len": random_stats["mean_ep_len"],
                    "random/win_rate_p0": random_stats["win_rate_p0"],
                    "random/loss_rate_p0": random_stats["loss_rate_p0"],
                    "random/draw_rate": random_stats["draw_rate"],
                }
            )

        # Sample game trajectory (text) occasionally
        if (update % TRAJ_LOG_INTERVAL == 0) and len(ep_traces) > 0:
            sample_ep = ep_traces[0]  # could random.choice if you prefer
            lines = [
                "Result P0={:.1f}, P1={:.1f}".format(
                    sample_ep["result_p0"], sample_ep["result_p1"]
                )
            ]
            for i, step in enumerate(sample_ep["steps"][:60]):
                lines.append(
                    f"{i:02d} | P{step['player']} | "
                    f"nsim={SIM_A if step['action'] == 0 else SIM_B} | "
                    f"my_t={step['my_time']:.2f} opp_t={step['opp_time']:.2f}"
                )
            traj_text = "\n".join(lines)
            log_dict["debug/sample_game"] = traj_text

        wandb.log(log_dict, step=update)

        if update % 10 == 0:
            print(
                f"[Update {update}] "
                f"loss={metrics_avg['loss']:.4f}  "
                f"policy={metrics_avg['policy_loss']:.4f}  "
                f"value={metrics_avg['value_loss']:.4f}  "
                f"entropy={metrics_avg['entropy']:.4f}  "
                f"KL={metrics_avg['approx_kl']:.6f}  "
                f"mean_ep_return={roll_stats['mean_ep_return']:.3f}  "
                f"mean_ep_len={roll_stats['mean_ep_len']:.1f}  "
                f"rate_8={roll_stats['action_rate_8']:.3f}  "
                f"rate_32={roll_stats['action_rate_32']:.3f}"
            )


if __name__ == "__main__":
    main()
