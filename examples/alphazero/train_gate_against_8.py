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

from speed_gardner_chess import (
    GardnerChess,
    MAX_TERMINATION_STEPS,
    _step_board,   # pure board step for MCTS
    _observe,      # board-only observation
    DEFAULT_TIME,  # 300 ticks per player
)
from network import AZNet


# ================================================================
# 1. Configs / hyperparams
# ================================================================

CKPT_ROOT = "/n/home04/amuppidi/speedchess/examples/alphazero/checkpoints"
BASE_ENV_ID = "gardner_chess"      # non-speed env used for training base models
ITER_FILE = "000400.ckpt"          # which checkpoint to use for each nsim_*
BASE_NSIMS = [8, 32]               # gate chooses between nsim=8 and nsim=32

SEED = 0

# PPO hyperparams
NUM_UPDATES = 3000                 # training iterations
ROLLOUT_STEPS = 2048               # TOTAL gating decisions per update (across envs)
GAMMA = 0.99
LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_CLIP_EPS = 0.2
PPO_LR = 3e-4
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.01
BATCH_SIZE = 256                   # minibatch size for PPO

# Number of parallel envs (vmap)
NUM_ENVS = 1024                      # tune this for your hardware
assert ROLLOUT_STEPS % NUM_ENVS == 0, "ROLLOUT_STEPS must be divisible by NUM_ENVS"
STEPS_PER_ENV = ROLLOUT_STEPS // NUM_ENVS

# Tick costs = number of MCTS simulations (used as "time spent" on clock)
COST_8 = 8
COST_32 = 32

# How often to log a sample game trajectory to wandb
TRAJ_LOG_INTERVAL = 50

# Gating checkpoint directory
GATE_CKPT_DIR = f"gate_checkpoints_8s_{NUM_ENVS}"

# ================================================================
# NEW: gate controls only Player 0; Player 1 = fixed AZ(8 sims)
# ================================================================
GATE_PLAYER = 0  # the learning gate controls P0 only


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
        policy_out, value_out = net(
            x, is_training=not is_eval, test_local_stats=False
        )
        return policy_out, value_out

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

        (logits, value), _ = forward.apply(
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
        (logits, value), _ = forward.apply(
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
# 4. GateNet: shared gating policy over {nsim_8, nsim_32} + value
# ================================================================

class GateNet(hk.Module):
    def __init__(self, num_options: int = 2, name: str = "GateNet"):
        super().__init__(name=name)
        self.num_options = num_options

    def __call__(self, obs, time_left_norm):
        """
        obs: (B, 5, 5, 115)   board observation from side-to-move perspective
        time_left_norm: (B, 2) = [my_time, opp_time] / default_time
        """
        x = obs.astype(jnp.float32)           # (B,5,5,115)
        x = jnp.moveaxis(x, -1, 1)            # (B,115,5,5)

        # Small conv trunk
        conv = hk.Sequential([
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Flatten(),
        ])
        z = conv(x)                           # (B, hidden)

        # Concatenate normalized times
        z = jnp.concatenate([z, time_left_norm], axis=-1)  # (B, hidden+2)

        h = hk.Linear(128)(z)
        h = jax.nn.relu(h)

        logits = hk.Linear(self.num_options)(h)            # (B,2)
        value = hk.Linear(1)(h)[..., 0]                    # (B,)
        return logits, value


def gate_forward_fn(obs_batch, time_batch):
    net = GateNet(num_options=2)
    return net(obs_batch, time_batch)


gate_forward = hk.without_apply_rng(hk.transform(gate_forward_fn))


# ================================================================
# 5. PPO loss, GAE helper
# ================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class PPOBatch:
    obs: jnp.ndarray         # (T*, 5,5,115)
    time: jnp.ndarray        # (T*, 2) normalized times
    actions: jnp.ndarray     # (T*,)
    logp_old: jnp.ndarray    # (T*,)
    values_old: jnp.ndarray  # (T*,)
    returns: jnp.ndarray     # (T*,)
    advantages: jnp.ndarray  # (T*,)
    # === NEW: mask indicating steps where gate actually controls the move (P0 turns)
    mask: jnp.ndarray        # (T*,)

    # Tell JAX how to treat this as a pytree
    def tree_flatten(self):
        children = (
            self.obs,
            self.time,
            self.actions,
            self.logp_old,
            self.values_old,
            self.returns,
            self.advantages,
            self.mask,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obs, time, actions, logp_old, values_old, returns, advantages, mask = children
        return cls(
            obs=obs,
            time=time,
            actions=actions,
            logp_old=logp_old,
            values_old=values_old,
            returns=returns,
            advantages=advantages,
            mask=mask,
        )


@jax.jit
def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones: (T, B)
    Returns: returns, advantages (both (T,B))
    """
    # Reverse for backward scan
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
            params, batch.obs, batch.time
        )  # logits: (T*,2), values: (T*,)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        logp = log_probs[jnp.arange(batch.actions.shape[0]), batch.actions]  # (T*,)

        # Ratio
        ratios = jnp.exp(logp - batch.logp_old)

        # === NEW: mask only gate-controlled states (P0 moves) for policy/entropy/kl
        mask = batch.mask
        mask_sum = jnp.maximum(1.0, jnp.sum(mask))

        # Normalize advantages over *gate* steps only
        adv_mean = jnp.sum(batch.advantages * mask) / mask_sum
        adv_var = jnp.sum(((batch.advantages - adv_mean) ** 2) * mask) / mask_sum
        adv_std = jnp.sqrt(adv_var + 1e-8)
        adv_norm = (batch.advantages - adv_mean) / adv_std

        # Clipped surrogate objective (only gate steps contribute)
        unclipped = ratios * adv_norm
        clipped = jnp.clip(ratios, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_norm
        policy_loss = -jnp.sum(jnp.minimum(unclipped, clipped) * mask) / mask_sum

        # Value loss: we can train on *all* states (P0 and P1 turns),
        # since value is "from state" (side-to-move perspective).
        value_loss = jnp.mean((batch.returns - values) ** 2)

        # Entropy: only gate-controlled states matter for exploration
        probs = jnp.exp(log_probs)
        entropy_per_state = -jnp.sum(probs * log_probs, axis=-1)
        entropy = jnp.sum(entropy_per_state * mask) / mask_sum

        loss = (
            policy_loss
            + PPO_VF_COEF * value_loss
            - PPO_ENT_COEF * entropy
        )

        approx_kl = 0.5 * jnp.sum(((logp - batch.logp_old) ** 2) * mask) / mask_sum
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
# 6. Helper: build PPOBatch + rollout stats from JAX traj
# ================================================================

def build_batch_and_stats(traj: Dict[str, jnp.ndarray]):
    """
    traj keys (T, B, ...):
      obs:    (T,B,5,5,115)
      time:   (T,B,2)
      action: (T,B)
      logp:   (T,B)
      value:  (T,B)
      reward: (T,B)   # from P0 perspective (gate)
      done:   (T,B)
      player: (T,B)
      mask:   (T,B)   # 1 if gate moved (P0), 0 if baseline moved (P1)
    """
    obs_arr = traj["obs"]        # (T,B,5,5,115)
    time_arr = traj["time"]      # (T,B,2)
    actions_arr = traj["action"] # (T,B)
    logp_arr = traj["logp"]      # (T,B)
    values_arr = traj["value"]   # (T,B)
    rewards_arr = traj["reward"] # (T,B)   # P0 reward signal
    dones_arr = traj["done"]     # (T,B)
    players_arr = traj["player"] # (T,B)
    mask_arr = traj["mask"]      # (T,B)

    T, B = actions_arr.shape

    # Compute returns and advantages (on device)
    returns, advantages = compute_gae(
        rewards_arr,
        values_arr,
        dones_arr.astype(jnp.float32),
        GAMMA,
        LAMBDA,
    )  # (T,B)

    # Flatten (T,B) -> (T*,)
    T_star = T * B
    obs_flat = obs_arr.reshape(T_star, 5, 5, 115)
    time_flat = time_arr.reshape(T_star, 2)
    actions_flat = actions_arr.reshape(T_star)
    logp_flat = logp_arr.reshape(T_star)
    values_flat = values_arr.reshape(T_star)
    returns_flat = returns.reshape(T_star)
    advantages_flat = advantages.reshape(T_star)
    mask_flat = mask_arr.reshape(T_star)

    batch = PPOBatch(
        obs=obs_flat,
        time=time_flat,
        actions=actions_flat,
        logp_old=logp_flat,
        values_old=values_flat,
        returns=returns_flat,
        advantages=advantages_flat,
        mask=mask_flat,
    )

    # ---------- Stats on host ----------
    actions_np = np.array(actions_arr, dtype=np.int32)   # (T,B)
    players_np = np.array(players_arr, dtype=np.int32)   # (T,B)
    dones_np = np.array(dones_arr > 0.5, dtype=bool)     # (T,B)
    rewards_np = np.array(rewards_arr, dtype=np.float32) # (T,B)  # P0 rewards
    adv_np = np.array(advantages, dtype=np.float32)      # (T,B)
    ret_np = np.array(returns, dtype=np.float32)         # (T,B)
    time_np = np.array(time_arr, dtype=np.float32)       # (T,B,2)
    mask_np = np.array(mask_arr, dtype=np.float32)       # (T,B)

    # Episode stats per env
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
            # sum of P0 rewards across episode
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

    # Action usage stats (flatten over time & envs)
    actions_flat_np = actions_np.reshape(-1)
    players_flat_np = players_np.reshape(-1)
    total_steps = max(1, len(actions_flat_np))

    rate_8 = float((actions_flat_np == 0).sum() / total_steps)
    rate_32 = float((actions_flat_np == 1).sum() / total_steps)

    mask0 = players_flat_np == 0
    mask1 = players_flat_np == 1
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

    # Move counts
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

    # Advantage / return stats
    stats.update(
        {
            "advantages_mean": float(adv_np.mean()),
            "advantages_std": float(adv_np.std()),
            "returns_mean": float(ret_np.mean()),
            "returns_std": float(ret_np.std()),
        }
    )

    # ---------- Per-player W/L/D + avg final time on clock ----------
    # NOTE: rewards are from P0's perspective now.
    wins_p0 = losses_p0 = draws_p0 = 0
    wins_p1 = losses_p1 = draws_p1 = 0
    final_time0_list: List[float] = []
    final_time1_list: List[float] = []
    num_games_total = 0

    for b in range(B):
        done_idx = np.where(dones_np[:, b])[0]
        if len(done_idx) == 0:
            continue
        start = 0
        time0 = float(DEFAULT_TIME)
        time1 = float(DEFAULT_TIME)
        for idx in done_idx:
            # accumulate time usage within this episode
            for t in range(start, idx + 1):
                p = int(players_np[t, b])   # 0 (gate) or 1 (baseline)
                a = int(actions_np[t, b])
                cost = COST_8 if a == 0 else COST_32
                if p == 0:
                    time0 -= cost
                else:
                    time1 -= cost

            # final times at end of episode
            final_time0_list.append(time0)
            final_time1_list.append(time1)
            num_games_total += 1

            # outcome for players (P0 reward is rewards_np[idx, b])
            r = float(rewards_np[idx, b])  # P0 reward
            eps = 1e-6
            if abs(r) <= eps:
                draws_p0 += 1
                draws_p1 += 1
            elif r > 0:
                wins_p0 += 1
                losses_p1 += 1
            else:  # r < 0
                losses_p0 += 1
                wins_p1 += 1

            # reset clock for next episode on this env
            time0 = float(DEFAULT_TIME)
            time1 = float(DEFAULT_TIME)
            start = idx + 1

    if num_games_total > 0:
        stats.update(
            {
                "win_rate_p0": wins_p0 / num_games_total,
                "loss_rate_p0": losses_p0 / num_games_total,
                "draw_rate_p0": draws_p0 / num_games_total,
                "win_rate_p1": wins_p1 / num_games_total,
                "loss_rate_p1": losses_p1 / num_games_total,
                "draw_rate_p1": draws_p1 / num_games_total,
                "avg_final_time_p0": float(np.mean(final_time0_list)),
                "avg_final_time_p1": float(np.mean(final_time1_list)),
            }
        )
    else:
        stats.update(
            {
                "win_rate_p0": 0.0,
                "loss_rate_p0": 0.0,
                "draw_rate_p0": 0.0,
                "win_rate_p1": 0.0,
                "loss_rate_p1": 0.0,
                "draw_rate_p1": 0.0,
                "avg_final_time_p0": float(DEFAULT_TIME),
                "avg_final_time_p1": float(DEFAULT_TIME),
            }
        )

    # ---------- Reconstruct some episode traces for debug ----------
    ep_traces: List[dict] = []
    # just log from env 0 for simplicity
    done_idx_0 = np.where(dones_np[:, 0])[0]
    if len(done_idx_0) > 0:
        start = 0
        for idx in done_idx_0[:3]:  # at most a few episodes
            steps: List[dict] = []
            for t in range(start, idx + 1):
                steps.append(
                    {
                        "player": int(players_np[t, 0]),
                        "action": int(actions_np[t, 0]),
                        "my_time": float(time_np[t, 0, 0]),
                        "opp_time": float(time_np[t, 0, 1]),
                    }
                )
            # episode result from P0 perspective
            r = float(rewards_np[idx, 0])
            if r > 0:
                result_p0 = 1.0
                result_p1 = -1.0
            elif r < 0:
                result_p0 = -1.0
                result_p1 = 1.0
            else:
                result_p0 = 0.0
                result_p1 = 0.0

            ep_traces.append(
                {
                    "steps": steps,
                    "result_p0": float(result_p0),
                    "result_p1": float(result_p1),
                }
            )
            start = idx + 1

    return batch, stats, ep_traces


# ================================================================
# 7. Jitted rollout (scan over time, vmap over envs)
# ================================================================

def make_rollout_core(
    env_speed: GardnerChess,
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
):
    default_time_f32 = jnp.float32(default_time)

    def rollout_core(state, gate_params, rng_key, num_steps: int):
        """
        Jitted rollout vs baseline:
          - Player 0: gate chooses nsim in {8,32}
          - Player 1: fixed AZ(8 sims) baseline
        state: batched State with leading dim NUM_ENVS
        """

        num_envs = NUM_ENVS

        def env_reset(key):
            return env_speed.init(key)

        def step_fn(carry, _):
            state, rng = carry  # state is batched (B, ...)

            # RNG splits
            rng, key_reset, key_gate, key_mcts8_gate, key_mcts32_gate, key_mcts8_base = jax.random.split(rng, 6)

            # Reset envs that were terminal/truncated
            done_prev = state.terminated | state.truncated   # (B,)
            keys_reset = jax.random.split(key_reset, num_envs)

            def reset_one(s, k, d):
                # both branches return a State
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

            # Normalize times from side-to-move perspective
            cur_idx = cur[:, None]
            opp_idx = (1 - cur)[:, None]
            my_time = jnp.take_along_axis(time_left, cur_idx, axis=1)[:, 0]
            opp_time = jnp.take_along_axis(time_left, opp_idx, axis=1)[:, 0]

            time_norm = jnp.stack(
                [my_time / default_time_f32, opp_time / default_time_f32],
                axis=1,
            )  # (B,2)

            # -------- Gate decision (batched) --------
            logits, value = gate_forward.apply(gate_params, obs, time_norm)
            # logits: (B,2), value: (B,)

            log_probs = jax.nn.log_softmax(logits, axis=-1)
            gate_action_sample = jax.random.categorical(key_gate, logits, axis=-1)  # (B,)
            batch_idx = jnp.arange(num_envs)
            logp_sample = log_probs[batch_idx, gate_action_sample]                  # (B,)

            # mask: 1 where gate actually controls the move (P0 to move)
            is_gate_player = (cur == GATE_PLAYER)                      # (B,)
            mask_gate = is_gate_player.astype(jnp.float32)             # (B,)

            # For P0 turns: use sampled gate action; for P1 turns: action=0 (8 sims baseline)
            gate_action = jnp.where(is_gate_player, gate_action_sample, 0)  # (B,)
            logp = jnp.where(is_gate_player, logp_sample, 0.0)              # (B,)

            # -------- Execute move: gate (P0) vs baseline (P1) --------
            key_mcts_env8_gate = jax.random.split(key_mcts8_gate, num_envs)
            key_mcts_env32_gate = jax.random.split(key_mcts32_gate, num_envs)
            key_mcts_env8_base = jax.random.split(key_mcts8_base, num_envs)

            def run_mcts_for_env(s, gate_turn, g_action, k8_gate, k32_gate, k8_base):
                # s: State for single env
                s_b = jax.tree_util.tree_map(lambda x: x[None, ...], s)

                def run_gate(_):
                    # gate controls this move
                    def run8(_):
                        a = select_mcts_8(model_8, s_b, k8_gate)[0]
                        return a, jnp.int32(COST_8)

                    def run32(_):
                        a = select_mcts_32(model_32, s_b, k32_gate)[0]
                        return a, jnp.int32(COST_32)

                    return jax.lax.cond(
                        g_action == 0,
                        run8,
                        run32,
                        operand=None,
                    )

                def run_baseline(_):
                    # baseline: fixed AZ(8 sims)
                    a = select_mcts_8(model_8, s_b, k8_base)[0]
                    return a, jnp.int32(COST_8)

                return jax.lax.cond(
                    gate_turn,
                    run_gate,
                    run_baseline,
                    operand=None,
                )

            actions, time_spent = jax.vmap(
                run_mcts_for_env,
                in_axes=(0, 0, 0, 0, 0, 0),
                out_axes=(0, 0),
            )(state, is_gate_player, gate_action, key_mcts_env8_gate, key_mcts_env32_gate, key_mcts_env8_base)

            prev_player = state.current_player     # (B,)

            def step_one(env_state, a, t):
                return env_speed.step(env_state, (a, t))

            state_next = jax.vmap(step_one, in_axes=(0, 0, 0))(
                state, actions, time_spent
            )

            done = (state_next.terminated | state_next.truncated).astype(jnp.float32)  # (B,)

            # === CHANGED: reward from gate's perspective (P0) regardless of mover
            rewards_all = state_next.rewards                           # (B,2)
            reward_p0 = rewards_all[:, GATE_PLAYER]
            reward = jnp.where(done > 0.0, reward_p0, jnp.float32(0.0))

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
                "mask": mask_gate,
            }
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            step_fn, (state, rng_key), xs=None, length=num_steps
        )
        return state_final, rng_final, traj

    rollout_core_jit = jax.jit(rollout_core, static_argnums=(3,))
    return rollout_core_jit


# Random gating rollout (for baseline evaluation)
def make_rollout_core_random(
    env_speed: GardnerChess,
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
):
    default_time_f32 = jnp.float32(default_time)

    def rollout_core_random(state, rng_key, num_steps: int):
        num_envs = NUM_ENVS

        def env_reset(key):
            return env_speed.init(key)

        def step_fn(carry, _):
            state, rng = carry

            rng, key_reset, key_gate, key_mcts8_gate, key_mcts32_gate, key_mcts8_base = jax.random.split(rng, 6)

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

            # -------- Random gate decision ONLY on P0 turns --------
            is_gate_player = (cur == GATE_PLAYER)                      # (B,)
            mask_gate = is_gate_player.astype(jnp.float32)             # (B,)

            gate_action_sample = jax.random.randint(key_gate, (num_envs,), 0, 2)  # (B,)
            # For P0: random 0/1 ; for P1: always 0 (8 sims)
            gate_action = jnp.where(is_gate_player, gate_action_sample, 0)
            logp = jnp.where(is_gate_player,
                             jnp.log(jnp.full((num_envs,), 0.5, dtype=jnp.float32)),
                             0.0)
            value = jnp.zeros((num_envs,), dtype=jnp.float32)

            # -------- Execute move via chosen MCTS per env --------
            key_mcts_env8_gate = jax.random.split(key_mcts8_gate, num_envs)
            key_mcts_env32_gate = jax.random.split(key_mcts32_gate, num_envs)
            key_mcts_env8_base = jax.random.split(key_mcts8_base, num_envs)

            def run_mcts_for_env(s, gate_turn, g_action, k8_gate, k32_gate, k8_base):
                s_b = jax.tree_util.tree_map(lambda x: x[None, ...], s)

                def run_gate(_):
                    def run8(_):
                        a = select_mcts_8(model_8, s_b, k8_gate)[0]
                        return a, jnp.int32(COST_8)

                    def run32(_):
                        a = select_mcts_32(model_32, s_b, k32_gate)[0]
                        return a, jnp.int32(COST_32)

                    return jax.lax.cond(
                        g_action == 0,
                        run8,
                        run32,
                        operand=None,
                    )

                def run_baseline(_):
                    a = select_mcts_8(model_8, s_b, k8_base)[0]
                    return a, jnp.int32(COST_8)

                return jax.lax.cond(
                    gate_turn,
                    run_gate,
                    run_baseline,
                    operand=None,
                )

            actions, time_spent = jax.vmap(
                run_mcts_for_env,
                in_axes=(0, 0, 0, 0, 0, 0),
                out_axes=(0, 0),
            )(state, is_gate_player, gate_action, key_mcts_env8_gate, key_mcts_env32_gate, key_mcts_env8_base)

            prev_player = state.current_player     # (B,)

            def step_one(env_state, a, t):
                return env_speed.step(env_state, (a, t))

            state_next = jax.vmap(step_one, in_axes=(0, 0, 0))(
                state, actions, time_spent
            )

            done = (state_next.terminated | state_next.truncated).astype(jnp.float32)  # (B,)

            # Reward from P0 perspective
            rewards_all = state_next.rewards                           # (B,2)
            reward_p0 = rewards_all[:, GATE_PLAYER]
            reward = jnp.where(done > 0.0, reward_p0, jnp.float32(0.0))

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
                "mask": mask_gate,
            }
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            step_fn, (state, rng_key), xs=None, length=num_steps
        )
        return state_final, rng_final, traj

    rollout_core_random_jit = jax.jit(rollout_core_random, static_argnums=(2,))
    return rollout_core_random_jit


def rollout_vs_baseline(
    rollout_core_jit,
    state,
    gate_params,
    rng_key,
    num_steps: int,
):
    """
    Wrapper: calls jitted rollout_core vs fixed AZ(8) baseline,
    then builds PPOBatch + stats + episode traces.
    """
    state_final, rng_final, traj = rollout_core_jit(
        state, gate_params, rng_key, num_steps
    )

    batch, stats, ep_traces = build_batch_and_stats(traj)
    return state_final, rng_final, batch, stats, ep_traces


# ================================================================
# 8. Jitted PPO epochs/minibatches (no dynamic slices)
# ================================================================

def make_ppo_update_epochs_fn(optimizer, value_and_grad):
    @jax.jit
    def ppo_update_epochs(params, opt_state, batch: PPOBatch, rng_key):
        T_star = batch.obs.shape[0]
        num_minibatches = T_star // BATCH_SIZE

        def one_epoch(carry, _):
            params, opt_state, rng = carry
            rng, key_perm = jax.random.split(rng)
            # Permute all indices, then chunk into minibatches
            perm = jax.random.permutation(key_perm, T_star)
            perm = perm[: num_minibatches * BATCH_SIZE]
            perm = perm.reshape((num_minibatches, BATCH_SIZE))  # (M, BATCH_SIZE)

            def one_minibatch(inner_carry, mb_indices):
                params_inner, opt_state_inner = inner_carry
                # mb_indices: (BATCH_SIZE,) integer indices into batch dimension
                mb = PPOBatch(
                    obs=batch.obs[mb_indices],
                    time=batch.time[mb_indices],
                    actions=batch.actions[mb_indices],
                    logp_old=batch.logp_old[mb_indices],
                    values_old=batch.values_old[mb_indices],
                    returns=batch.returns[mb_indices],
                    advantages=batch.advantages[mb_indices],
                    mask=batch.mask[mb_indices],
                )
                (loss, metrics), grads = value_and_grad(params_inner, mb)
                updates, opt_state_inner = optimizer.update(
                    grads, opt_state_inner, params_inner
                )
                params_inner = optax.apply_updates(params_inner, updates)
                return (params_inner, opt_state_inner), metrics

            (params, opt_state), metrics_mb = jax.lax.scan(
                one_minibatch,
                (params, opt_state),
                perm,
            )
            metrics_epoch = {k: jnp.mean(v) for k, v in metrics_mb.items()}
            return (params, opt_state, rng), metrics_epoch

        (params, opt_state, rng_key), metrics_epochs = jax.lax.scan(
            one_epoch, (params, opt_state, rng_key), jnp.arange(PPO_EPOCHS)
        )
        metrics_final = {k: jnp.mean(v) for k, v in metrics_epochs.items()}
        return params, opt_state, rng_key, metrics_final

    return ppo_update_epochs


# ================================================================
# 9. Checkpoint helpers for gating policy
# ================================================================

def save_gate_checkpoint(path, gate_params, opt_state, update, roll_stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "update": int(update),
        "gate_params": jax.device_get(gate_params),
        "opt_state": jax.device_get(opt_state),
        "roll_stats": roll_stats,
    }
    with open(path, "wb") as f:
        pickle.dump(to_save, f)


# ================================================================
# 10. Main training loop (gate vs AZ(8) baseline) + wandb + tqdm
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

    env_id_8, cfg_8, model_8 = load_checkpoint(ckpt_paths["nsim_8"])
    env_id_32, cfg_32, model_32 = load_checkpoint(ckpt_paths["nsim_32"])
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

    # single env just to read default_time
    rng, key_single, key_batch = jax.random.split(rng, 3)
    single_state = env_speed.init(key_single)
    default_time = float(single_state.time_left[0])

    # Batched initial states
    init_keys = jax.random.split(key_batch, NUM_ENVS)
    state = jax.vmap(env_speed.init)(init_keys)  # batched State

    forward = build_forward(env_speed, cfg_8)  # same arch for both base models

    # ----------------------
    # MCTS selectors (shared nets for both players)
    # ----------------------
    recurrent_fn_speed = make_recurrent_fn_speed(forward)
    select_mcts_8 = make_select_actions_mcts(forward, recurrent_fn_speed, 8)
    select_mcts_32 = make_select_actions_mcts(forward, recurrent_fn_speed, 32)

    # ----------------------
    # Init GateNet + optimizer
    # ----------------------
    dummy_obs = jnp.zeros((1, 5, 5, 115), dtype=jnp.float32)
    dummy_time = jnp.zeros((1, 2), dtype=jnp.float32)
    gate_params = gate_forward.init(
        jax.random.PRNGKey(SEED + 123), dummy_obs, dummy_time
    )

    optimizer = optax.adam(PPO_LR)
    opt_state = optimizer.init(gate_params)

    loss_fn = make_ppo_loss_fn()
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # Jitted PPO epochs
    ppo_update_epochs = make_ppo_update_epochs_fn(optimizer, value_and_grad)

    # ----------------------
    # Jitted rollout cores
    # ----------------------
    rollout_core_jit = make_rollout_core(
        env_speed,
        select_mcts_8,
        select_mcts_32,
        model_8,
        model_32,
        default_time,
    )
    rollout_core_random_jit = make_rollout_core_random(
        env_speed,
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
        project="pgx-speed-gate-vmap",
        config={
            "ckpt_root": CKPT_ROOT,
            "base_env_id": BASE_ENV_ID,
            "iter_file": ITER_FILE,
            "base_nsims": BASE_NSIMS,
            "seed": SEED,
            "num_updates": NUM_UPDATES,
            "rollout_steps_total": ROLLOUT_STEPS,
            "steps_per_env": STEPS_PER_ENV,
            "num_envs": NUM_ENVS,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "ppo_epochs": PPO_EPOCHS,
            "ppo_clip_eps": PPO_CLIP_EPS,
            "ppo_lr": PPO_LR,
            "ppo_vf_coef": PPO_VF_COEF,
            "ppo_ent_coef": PPO_ENT_COEF,
            "batch_size": BATCH_SIZE,
            "cost_8": COST_8,
            "cost_32": COST_32,
            "gate_player": GATE_PLAYER,
            "baseline_policy": "AZ_MCTS_8_sims",
        },
        name="gate_speed_gardner_vs_az8_vmap",
    )
    print("initialized wandb and checkpoint loading")

    # Checkpoint schedule: every 10% of updates
    os.makedirs(GATE_CKPT_DIR, exist_ok=True)
    ckpt_interval = max(1, NUM_UPDATES // 2)
    best_mean_ep_return = -1e9

    # Save initial (update 0) gating params
    init_ckpt_path = os.path.join(GATE_CKPT_DIR, "gate_update_000000.pkl")
    save_gate_checkpoint(init_ckpt_path, gate_params, opt_state, 0, {"mean_ep_return": 0.0})

    # ----------------------
    # Training loop
    # ----------------------
    for update in trange(1, NUM_UPDATES + 1, desc="PPO updates"):
        # 1) Multi-env jitted rollout (gate vs AZ(8) baseline)
        state, rng, batch, roll_stats, ep_traces = rollout_vs_baseline(
            rollout_core_jit,
            state,
            gate_params,
            rng,
            STEPS_PER_ENV,
        )

        # 2) Jitted PPO over epochs & minibatches
        rng, ppo_rng = jax.random.split(rng)
        gate_params, opt_state, rng, metrics_avg_jax = ppo_update_epochs(
            gate_params, opt_state, batch, rng
        )
        metrics_avg = {k: float(v) for k, v in jax.device_get(metrics_avg_jax).items()}

        # 3) Possibly run random-gating evaluation + checkpoints every 10%
        log_dict: Dict[str, Any] = {}

        if update % ckpt_interval == 0:
            # --- checkpoint gating params ---
            ckpt_path = os.path.join(GATE_CKPT_DIR, f"gate_update_{update:06d}.pkl")
            save_gate_checkpoint(ckpt_path, gate_params, opt_state, update, roll_stats)

            # --- best gating policy by mean_ep_return ---
            mean_ret = roll_stats["mean_ep_return"]
            if mean_ret > best_mean_ep_return:
                best_mean_ep_return = mean_ret
                best_ckpt_path = os.path.join(GATE_CKPT_DIR, "gate_best.pkl")
                save_gate_checkpoint(best_ckpt_path, gate_params, opt_state, update, roll_stats)
                log_dict["checkpoint/best_update"] = update
                log_dict["checkpoint/best_mean_ep_return"] = mean_ret

            # --- evaluation vs random gating policy (P0 random gate vs AZ8 baseline) ---
            rng, eval_state_key, eval_rng = jax.random.split(rng, 3)
            eval_init_keys = jax.random.split(eval_state_key, NUM_ENVS)
            eval_state = jax.vmap(env_speed.init)(eval_init_keys)
            eval_state, eval_rng_out, traj_random = rollout_core_random_jit(
                eval_state, eval_rng, STEPS_PER_ENV
            )
            _batch_rand, rand_stats, _ep_traces_rand = build_batch_and_stats(traj_random)

            # Log random-gating stats
            for k, v in rand_stats.items():
                log_dict[f"eval_random/{k}"] = v

        # Build wandb log dict for training rollout / losses
        log_dict.update(
            {
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
                # per-player W/L/D + avg final clock time
                "rollout/win_rate_p0": roll_stats["win_rate_p0"],
                "rollout/loss_rate_p0": roll_stats["loss_rate_p0"],
                "rollout/draw_rate_p0": roll_stats["draw_rate_p0"],
                "rollout/win_rate_p1": roll_stats["win_rate_p1"],
                "rollout/loss_rate_p1": roll_stats["loss_rate_p1"],
                "rollout/draw_rate_p1": roll_stats["draw_rate_p1"],
                "rollout/avg_final_time_p0": roll_stats["avg_final_time_p0"],
                "rollout/avg_final_time_p1": roll_stats["avg_final_time_p1"],
            }
        )

        # 4) Sample game: number of moves + nsim-per-move plot
        if (update % TRAJ_LOG_INTERVAL == 0) and len(ep_traces) > 0:
            sample_ep = ep_traces[0]  # first episode from env 0
            steps = sample_ep["steps"]
            num_moves = len(steps)
            log_dict["debug/sample_game_num_moves"] = num_moves

            # Text dump as before
            lines = [
                "Result P0={:.1f}, P1={:.1f}".format(
                    sample_ep["result_p0"], sample_ep["result_p1"]
                )
            ]
            for i, step in enumerate(steps[:60]):
                lines.append(
                    f"{i:02d} | P{step['player']} | "
                    f"nsim={8 if step['action'] == 0 else 32} | "
                    f"my_t={step['my_time']:.2f} opp_t={step['opp_time']:.2f}"
                )
            traj_text = "\n".join(lines)
            log_dict["debug/sample_game"] = traj_text

            # W&B table + line plot: move index vs nsim used
            table = wandb.Table(columns=["move", "nsim", "player", "my_time", "opp_time"])
            for i, step in enumerate(steps):
                nsim = 8 if step["action"] == 0 else 32
                table.add_data(
                    i,
                    nsim,
                    int(step["player"]),
                    float(step["my_time"]),
                    float(step["opp_time"]),
                )
            nsim_plot = wandb.plot.line(
                table,
                x="move",
                y="nsim",
                title="nsim choice per move (sample game)",
            )
            log_dict["debug/sample_game_nsim_per_move"] = nsim_plot

        # 5) Log to wandb
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
