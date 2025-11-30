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
    _step_board,   # pure board step for MCTS
    _observe,      # board-only observation
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

# Multi-env rollout parameters
NUM_ENVS = 32                      # parallel speed-chess envs

# PPO hyperparams
NUM_UPDATES = 1000                 # training iterations
ROLLOUT_STEPS_TOTAL = 2048         # total gating decisions per PPO update (across all envs)
assert ROLLOUT_STEPS_TOTAL % NUM_ENVS == 0, "ROLLOUT_STEPS_TOTAL must be divisible by NUM_ENVS"
STEPS_PER_ENV = ROLLOUT_STEPS_TOTAL // NUM_ENVS  # time steps per env per update

GAMMA = 0.99
LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_CLIP_EPS = 0.2
PPO_LR = 3e-4
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.01
BATCH_SIZE = 256                   # minibatch size for PPO

# Tick costs = number of MCTS simulations
COST_8 = 8
COST_32 = 32

# How often to log a sample game trajectory to wandb
TRAJ_LOG_INTERVAL = 50


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

        conv = hk.Sequential([
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Flatten(),
        ])
        z = conv(x)                           # (B, hidden)

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

    def tree_flatten(self):
        children = (
            self.obs,
            self.time,
            self.actions,
            self.logp_old,
            self.values_old,
            self.returns,
            self.advantages,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obs, time, actions, logp_old, values_old, returns, advantages = children
        return cls(
            obs=obs,
            time=time,
            actions=actions,
            logp_old=logp_old,
            values_old=values_old,
            returns=returns,
            advantages=advantages,
        )


@jax.jit
def compute_gae_1d(rewards, values, dones, gamma, lam):
    """
    1D GAE:
      rewards, values, dones: (T,)
    Returns: returns, advantages
    """
    rev_rewards = rewards[::-1]
    rev_values = values[::-1]
    rev_dones = dones[::-1]

    def body_fn(carry, inp):
        next_value, next_adv = carry
        r, v, d = inp
        mask = 1.0 - d
        delta = r + gamma * next_value * mask - v
        adv = delta + gamma * lam * mask * next_adv
        ret = adv + v
        return (v, adv), (ret, adv)

    (_, _), (rev_returns, rev_advantages) = jax.lax.scan(
        body_fn,
        (0.0, 0.0),
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

        ratios = jnp.exp(logp - batch.logp_old)

        adv_mean = jnp.mean(batch.advantages)
        adv_std = jnp.std(batch.advantages) + 1e-8
        adv_norm = (batch.advantages - adv_mean) / adv_std

        unclipped = ratios * adv_norm
        clipped = jnp.clip(ratios, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_norm
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        value_loss = jnp.mean((batch.returns - values) ** 2)

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
# 6. Helper: build PPOBatch + rollout stats from JAX traj
# ================================================================

def build_batch_and_stats(traj: Dict[str, jnp.ndarray]):
    """
    traj keys (T_env, B, ...):
      obs:    (T_env,B,5,5,115)
      time:   (T_env,B,2)
      action: (T_env,B)
      logp:   (T_env,B)
      value:  (T_env,B)
      reward: (T_env,B)
      done:   (T_env,B)
      player: (T_env,B)
    """
    obs_arr = traj["obs"]        # (T,B,5,5,115)
    time_arr = traj["time"]      # (T,B,2)
    actions_arr = traj["action"] # (T,B)
    logp_arr = traj["logp"]      # (T,B)
    values_arr = traj["value"]   # (T,B)
    rewards_arr = traj["reward"] # (T,B)
    dones_arr = traj["done"]     # (T,B)
    players_arr = traj["player"] # (T,B)

    T_env, B = actions_arr.shape

    # --------- GAE per env (on device) ---------
    dones_f = dones_arr.astype(jnp.float32)

    def gae_per_env(r, v, d):
        return compute_gae_1d(r, v, d, GAMMA, LAMBDA)

    returns_arr, advantages_arr = jax.vmap(
        gae_per_env,
        in_axes=(1, 1, 1),
        out_axes=(1, 1),
    )(rewards_arr, values_arr, dones_f)  # (T_env,B)

    # --------- Flatten for PPO (T* = T_env * B) ---------
    T_star = T_env * B
    obs_flat = obs_arr.reshape(T_star, 5, 5, 115)
    time_flat = time_arr.reshape(T_star, 2)
    actions_flat = actions_arr.reshape(T_star)
    logp_flat = logp_arr.reshape(T_star)
    values_flat = values_arr.reshape(T_star)
    returns_flat = returns_arr.reshape(T_star)
    advantages_flat = advantages_arr.reshape(T_star)

    batch = PPOBatch(
        obs=obs_flat,
        time=time_flat,
        actions=actions_flat,
        logp_old=logp_flat,
        values_old=values_flat,
        returns=returns_flat,
        advantages=advantages_flat,
    )

    # ---------- Stats on host ----------
    rewards_np = np.array(rewards_arr, dtype=np.float32)   # (T_env,B)
    dones_np = np.array(dones_arr > 0.5, dtype=bool)       # (T_env,B)
    actions_np = np.array(actions_arr, dtype=np.int32)     # (T_env,B)
    players_np = np.array(players_arr, dtype=np.int32)     # (T_env,B)
    adv_np = np.array(advantages_arr, dtype=np.float32)    # (T_env,B)
    ret_np = np.array(returns_arr, dtype=np.float32)       # (T_env,B)
    time_np = np.array(time_arr, dtype=np.float32)         # (T_env,B,2)

    # Episode stats per env
    ep_lens: List[int] = []
    ep_returns: List[float] = []

    for b in range(B):
        done_idx = np.where(dones_np[:, b])[0]
        if len(done_idx) == 0:
            continue
        start = 0
        for idx in done_idx:
            length = idx - start + 1
            ep_lens.append(length)
            ep_returns.append(float(rewards_np[start:idx + 1, b].sum()))
            start = idx + 1

    if len(ep_lens) == 0:
        ep_lens_arr = np.array([T_env], dtype=np.int32)
        ep_returns_arr = np.array([rewards_np.sum()], dtype=np.float32)
    else:
        ep_lens_arr = np.array(ep_lens, dtype=np.int32)
        ep_returns_arr = np.array(ep_returns, dtype=np.float32)

    stats: Dict[str, float] = {
        "mean_ep_return": float(ep_returns_arr.mean()),
        "mean_ep_len": float(ep_lens_arr.mean()),
    }

    # Action usage stats (flatten over time & envs)
    actions_flat_np = actions_np.reshape(-1)
    players_flat_np = players_np.reshape(-1)
    total_steps = max(1, actions_flat_np.shape[0])

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

    num_games = max(1, len(ep_lens_arr))
    moves_p0 = int((players_flat_np == 0).sum())
    moves_p1 = int((players_flat_np == 1).sum())
    avg_moves_per_game = float(ep_lens_arr.mean())
    stats.update(
        {
            "avg_moves_per_game": avg_moves_per_game,
            "avg_moves_p0": float(moves_p0) / num_games,
            "avg_moves_p1": float(moves_p1) / num_games,
        }
    )

    stats.update(
        {
            "advantages_mean": float(adv_np.mean()),
            "advantages_std": float(adv_np.std()),
            "returns_mean": float(ret_np.mean()),
            "returns_std": float(ret_np.std()),
        }
    )

    # ---------- Reconstruct episode traces for env 0 ----------
    ep_traces: List[dict] = []
    dones0 = dones_np[:, 0]
    rewards0 = rewards_np[:, 0]
    players0 = players_np[:, 0]
    time0 = time_np[:, 0, :]
    actions0 = actions_np[:, 0]

    done_idx0 = np.where(dones0)[0]
    if len(done_idx0) > 0:
        start = 0
        for idx in done_idx0[:3]:
            steps: List[dict] = []
            for t in range(start, idx + 1):
                steps.append(
                    {
                        "player": int(players0[t]),
                        "action": int(actions0[t]),
                        "my_time": float(time0[t, 0]),
                        "opp_time": float(time0[t, 1]),
                    }
                )
            r = float(rewards0[idx])
            mover = int(players0[idx])
            if r > 0:
                if mover == 0:
                    result_p0, result_p1 = 1.0, -1.0
                else:
                    result_p0, result_p1 = -1.0, 1.0
            elif r < 0:
                if mover == 0:
                    result_p0, result_p1 = -1.0, 1.0
                else:
                    result_p0, result_p1 = 1.0, -1.0
            else:
                result_p0 = result_p1 = 0.0

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
        Jitted self-play rollout using jax.lax.scan over time
        and vmap over NUM_ENVS parallel environments.

        state: batched GardnerChess.State with leading axis NUM_ENVS
        """
        batch_size = state.current_player.shape[0]
        assert batch_size == NUM_ENVS, "state batch size must equal NUM_ENVS"

        def one_step(carry, _):
            state, rng = carry

            rng, key_reset, key_gate, key_mcts = jax.random.split(rng, 4)

            # Reset envs that were terminal/truncated
            done_prev = state.terminated | state.truncated  # (N,)

            reset_keys = jax.random.split(key_reset, NUM_ENVS)
            reset_states = jax.vmap(env_speed.init)(reset_keys)

            def mix_leaf(old, new):
                if old.ndim == 0:
                    return old
                expand = (1,) * (old.ndim - 1)
                mask = done_prev.reshape((-1,) + expand)
                return jnp.where(mask, new, old)

            state = jax.tree_util.tree_map(mix_leaf, state, reset_states)

            # Side-to-move observation & times
            obs = state.observation               # (N,5,5,115)
            time_left = state.time_left           # (N,2)
            cur = state.current_player            # (N,)

            idxs = jnp.arange(NUM_ENVS)
            my_time = time_left[idxs, cur]
            opp_time = time_left[idxs, 1 - cur]
            time_norm = jnp.stack(
                [my_time / default_time_f32, opp_time / default_time_f32],
                axis=1,
            )                                     # (N,2)

            # -------- Gate decision (batched) --------
            logits, value = gate_forward.apply(gate_params, obs, time_norm)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            gate_actions = jax.random.categorical(key_gate, logits, axis=-1)  # (N,)
            logp = log_probs[idxs, gate_actions]                               # (N,)

            # -------- Execute move via chosen MCTS per env --------
            key_mcts_env = jax.random.split(key_mcts, NUM_ENVS)

            def select_action_for_env(env_state, gate_action, k_mcts):
                env_state_b = jax.tree_util.tree_map(lambda x: x[None, ...], env_state)

                def run_8(_):
                    a = select_mcts_8(model_8, env_state_b, k_mcts)[0]
                    return a, jnp.int32(COST_8)

                def run_32(_):
                    a = select_mcts_32(model_32, env_state_b, k_mcts)[0]
                    return a, jnp.int32(COST_32)

                return jax.lax.cond(
                    gate_action == 0,
                    run_8,
                    run_32,
                    operand=None,
                )

            actions, time_spent = jax.vmap(
                select_action_for_env,
                in_axes=(0, 0, 0),
                out_axes=(0, 0),
            )(state, gate_actions, key_mcts_env)  # (N,), (N,)

            prev_player = state.current_player     # (N,)

            def step_one(env_state, a, ts):
                return env_speed.step(env_state, (a, ts))

            state_next = jax.vmap(step_one, in_axes=(0, 0, 0))(
                state, actions, time_spent
            )

            done_after = state_next.terminated | state_next.truncated  # (N,)
            rewards_all = state_next.rewards                           # (N,2)
            prev_idx = prev_player[:, None]
            reward_prev = jnp.take_along_axis(rewards_all, prev_idx, axis=1)[:, 0]
            reward = jnp.where(done_after, reward_prev, jnp.float32(0.0))  # (N,)

            out = {
                "obs": obs,
                "time": time_norm,
                "action": gate_actions,
                "logp": logp,
                "value": value,
                "reward": reward,
                "done": done_after,
                "player": prev_player,
            }

            carry_next = (state_next, rng)
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            one_step, (state, rng_key), xs=None, length=num_steps
        )
        return state_final, rng_final, traj

    rollout_core_jit = jax.jit(rollout_core, static_argnums=(3,))
    return rollout_core_jit


def rollout_selfplay(
    rollout_core_jit,
    state,
    gate_params,
    rng_key,
    num_steps: int,
):
    """
    Wrapper: calls jitted rollout_core, then builds PPOBatch + stats + episode traces.
    """
    state_final, rng_final, traj = rollout_core_jit(
        state, gate_params, rng_key, num_steps
    )

    batch, stats, ep_traces = build_batch_and_stats(traj)
    return state_final, rng_final, batch, stats, ep_traces


# ================================================================
# 8. Jitted PPO epochs/minibatches (with dynamic_slice_in_dim)
# ================================================================

def make_ppo_update_epochs_fn(optimizer, value_and_grad):
    @jax.jit
    def ppo_update_epochs(params, opt_state, batch: PPOBatch, rng_key):
        T_star = batch.obs.shape[0]
        num_minibatches = T_star // BATCH_SIZE

        def one_epoch(carry, _):
            params, opt_state, rng = carry
            rng, key_perm = jax.random.split(rng)
            idxs = jax.random.permutation(key_perm, T_star)

            perm_batch = PPOBatch(
                obs=batch.obs[idxs],
                time=batch.time[idxs],
                actions=batch.actions[idxs],
                logp_old=batch.logp_old[idxs],
                values_old=batch.values_old[idxs],
                returns=batch.returns[idxs],
                advantages=batch.advantages[idxs],
            )

            def slice_minibatch(x, start):
                # x: (T_star, ...)
                return jax.lax.dynamic_slice_in_dim(
                    x, start_index=start, slice_size=BATCH_SIZE, axis=0
                )

            def one_minibatch(inner_carry, mb_idx):
                params, opt_state = inner_carry
                start = mb_idx * BATCH_SIZE

                mb = PPOBatch(
                    obs=slice_minibatch(perm_batch.obs, start),
                    time=slice_minibatch(perm_batch.time, start),
                    actions=slice_minibatch(perm_batch.actions, start),
                    logp_old=slice_minibatch(perm_batch.logp_old, start),
                    values_old=slice_minibatch(perm_batch.values_old, start),
                    returns=slice_minibatch(perm_batch.returns, start),
                    advantages=slice_minibatch(perm_batch.advantages, start),
                )

                (loss, metrics), grads = value_and_grad(params, mb)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics

            (params, opt_state), metrics_mb = jax.lax.scan(
                one_minibatch,
                (params, opt_state),
                jnp.arange(num_minibatches),
            )

            metrics_epoch = {k: jnp.mean(v) for k, v in metrics_mb.items()}
            return (params, opt_state, rng), metrics_epoch

        (params, opt_state, rng_key_out), metrics_epochs = jax.lax.scan(
            one_epoch, (params, opt_state, rng_key), jnp.arange(PPO_EPOCHS)
        )
        metrics_final = {k: jnp.mean(v) for k, v in metrics_epochs.items()}
        return params, opt_state, rng_key_out, metrics_final

    return ppo_update_epochs


# ================================================================
# 9. Main training loop (self-play PPO) + wandb + tqdm
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
    rng, key_single, key_batch = jax.random.split(rng, 3)

    # single env just to read default_time
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
    # Jitted rollout core
    # ----------------------
    rollout_core_jit = make_rollout_core(
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
            "rollout_steps_total": ROLLOUT_STEPS_TOTAL,
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
        },
        name="gate_speed_gardner_selfplay_vmap",
    )
    print("initialized wandb and checkpoint loading")

    # ----------------------
    # Training loop
    # ----------------------
    for update in trange(1, NUM_UPDATES + 1, desc="PPO updates"):
        # 1) Multi-env jitted rollout
        state, rng, batch, roll_stats, ep_traces = rollout_selfplay(
            rollout_core_jit,
            state,
            gate_params,
            rng,
            STEPS_PER_ENV,
        )

        # 2) Jitted PPO epochs × minibatches
        rng, ppo_rng = jax.random.split(rng)
        gate_params, opt_state, rng, metrics_avg_jax = ppo_update_epochs(
            gate_params, opt_state, batch, rng
        )
        metrics_avg = {k: float(v) for k, v in metrics_avg_jax.items()}

        # Build wandb log dict
        log_dict = {
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
        }

        # Sample game trajectory (text) occasionally
        if (update % TRAJ_LOG_INTERVAL == 0) and len(ep_traces) > 0:
            sample_ep = ep_traces[0]
            lines = [
                "Result P0={:.1f}, P1={:.1f}".format(
                    sample_ep["result_p0"], sample_ep["result_p1"]
                )
            ]
            for i, step in enumerate(sample_ep["steps"][:60]):
                lines.append(
                    f"{i:02d} | P{step['player']} | "
                    f"nsim={8 if step['action'] == 0 else 32} | "
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
