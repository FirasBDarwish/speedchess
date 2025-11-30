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

# Tick costs = number of MCTS simulations
COST_8 = 8
COST_32 = 32

# How often to log a sample game trajectory to wandb
TRAJ_LOG_INTERVAL = 50

# Gate checkpoint directory
GATE_CKPT_ROOT = "./gate_checkpoints_j2"


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
    obs: jnp.ndarray         # (T or U, B, 5,5,115) or (T,5,5,115)
    time: jnp.ndarray        # (T or U, B, 2) or (T,2)
    actions: jnp.ndarray     # (T or U, B) or (T,)
    logp_old: jnp.ndarray    # (T or U, B) or (T,)
    values_old: jnp.ndarray  # (T or U, B) or (T,)
    returns: jnp.ndarray     # (T or U, B) or (T,)
    advantages: jnp.ndarray  # (T or U, B) or (T,)

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
def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones: (T,)
    Returns: returns, advantages
    """
    # Reverse for backward scan
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

        # Clipped surrogate objective
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
# 6. Helper: build PPOBatch + rollout stats + per-episode info
# ================================================================

def build_batch_and_stats(traj: Dict[str, jnp.ndarray], default_time: float):
    """
    traj keys:
      obs:    (T,5,5,115)
      time:   (T,2)   -- normalized [my_time/default_time, opp_time/default_time]
      action: (T,)    -- 0 for nsim=8, 1 for nsim=32
      logp:   (T,)
      value:  (T,)
      reward: (T,)    -- nonzero only at terminal
      done:   (T,)
      player: (T,)    -- player who moved (0/1)
    """
    obs_arr = traj["obs"]
    time_arr = traj["time"]
    actions_arr = traj["action"]
    logp_arr = traj["logp"]
    values_arr = traj["value"]
    rewards_arr = traj["reward"]
    dones_arr = traj["done"]
    players_arr = traj["player"]

    # Compute returns and advantages (still on device)
    returns, advantages = compute_gae(
        rewards_arr,
        values_arr,
        dones_arr.astype(jnp.float32),
        GAMMA,
        LAMBDA,
    )

    batch = PPOBatch(
        obs=obs_arr,
        time=time_arr,
        actions=actions_arr,
        logp_old=logp_arr,
        values_old=values_arr,
        returns=returns,
        advantages=advantages,
    )

    # ---------- Stats + episode recon on host ----------
    actions_np = np.array(actions_arr, dtype=np.int32)
    players_np = np.array(players_arr, dtype=np.int32)
    dones_np = np.array(dones_arr > 0.5, dtype=bool)
    rewards_np = np.array(rewards_arr, dtype=np.float32)
    adv_np = np.array(advantages, dtype=np.float32)
    ret_np = np.array(returns, dtype=np.float32)
    time_np = np.array(time_arr, dtype=np.float32)

    T = actions_np.shape[0]

    # Episode boundaries: indices where done=True
    done_idx = np.where(dones_np)[0]
    if len(done_idx) == 0:
        ep_lens = np.array([T], dtype=np.int32)
        ep_returns = np.array([rewards_np.sum()], dtype=np.float32)
    else:
        ep_lens = np.diff(np.concatenate([[-1], done_idx])).astype(np.int32)
        ep_returns_list: List[float] = []
        start = 0
        for idx in done_idx:
            ep_returns_list.append(float(rewards_np[start:idx + 1].sum()))
            start = idx + 1
        ep_returns = np.array(ep_returns_list, dtype=np.float32)

    stats: Dict[str, float] = {
        "mean_ep_return": float(ep_returns.mean()),
        "mean_ep_len": float(ep_lens.mean()),
    }

    # Action usage stats
    total_steps = max(1, len(actions_np))
    rate_8 = float((actions_np == 0).sum() / total_steps)
    rate_32 = float((actions_np == 1).sum() / total_steps)

    mask0 = players_np == 0
    mask1 = players_np == 1
    total_p0 = max(1, int(mask0.sum()))
    total_p1 = max(1, int(mask1.sum()))

    rate_8_p0 = float((actions_np[mask0] == 0).sum() / total_p0) if total_p0 > 0 else 0.0
    rate_8_p1 = float((actions_np[mask1] == 0).sum() / total_p1) if total_p1 > 0 else 0.0

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
    moves_p0 = int(mask0.sum())
    moves_p1 = int(mask1.sum())
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

    # ---------- Reconstruct episode traces + win/loss/draw + time + per-move nsim ----------
    ep_traces: List[dict] = []
    hist_vals_8: List[int] = []
    hist_vals_32: List[int] = []

    wins_p0 = 0
    losses_p0 = 0
    draws = 0

    final_times_p0: List[float] = []
    final_times_p1: List[float] = []

    if len(done_idx) > 0:
        start = 0
        for idx in done_idx:
            steps: List[dict] = []

            # Reconstruct clock for this episode from default_time
            time_left = np.array([default_time, default_time], dtype=np.float32)

            for t in range(start, idx + 1):
                local_move_idx = t - start
                a = int(actions_np[t])

                # For hist: which move index used which nsim?
                if a == 0:
                    hist_vals_8.append(local_move_idx)
                    cost = COST_8
                else:
                    hist_vals_32.append(local_move_idx)
                    cost = COST_32

                p = int(players_np[t])
                time_left[p] -= cost

                steps.append(
                    {
                        "player": p,
                        "action": a,  # 0->8 sims, 1->32 sims
                        "my_time": float(time_np[t, 0]),   # normalized
                        "opp_time": float(time_np[t, 1]),  # normalized
                    }
                )

            # Clip clocks at >=0 (if time flag)
            final_times_p0.append(float(max(time_left[0], 0.0)))
            final_times_p1.append(float(max(time_left[1], 0.0)))

            # Final outcome from rewards: reward is for prev_player at termination
            r = float(rewards_np[idx])
            mover = int(players_np[idx])
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

            ep_traces.append(
                {
                    "steps": steps,
                    "result_p0": float(result_p0),
                    "result_p1": float(result_p1),
                }
            )
            start = idx + 1

        num_games = len(done_idx)
        stats.update(
            {
                "win_rate_p0": wins_p0 / num_games,
                "loss_rate_p0": losses_p0 / num_games,
                "draw_rate": draws / num_games,
                "avg_final_time_p0": float(np.mean(final_times_p0)),
                "avg_final_time_p1": float(np.mean(final_times_p1)),
                "avg_final_time_p0_norm": float(np.mean(final_times_p0)) / default_time,
                "avg_final_time_p1_norm": float(np.mean(final_times_p1)) / default_time,
            }
        )
    else:
        stats.update(
            {
                "win_rate_p0": 0.0,
                "loss_rate_p0": 0.0,
                "draw_rate": 0.0,
                "avg_final_time_p0": 0.0,
                "avg_final_time_p1": 0.0,
                "avg_final_time_p0_norm": 0.0,
                "avg_final_time_p1_norm": 0.0,
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
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
):
    default_time_f32 = jnp.float32(default_time)

    def rollout_core(state, gate_params, rng_key, num_steps: int):
        """
        Jitted self-play rollout using jax.lax.scan, GateNet gating.
        """

        def one_step(carry, _):
            state, rng = carry

            # Reset episode if previous state was terminal/truncated
            rng, key_reset, key_gate, key_mcts = jax.random.split(rng, 4)
            done_prev = state.terminated | state.truncated

            def reset_fn(carry_inner):
                _state, k = carry_inner
                return env_speed.init(k)

            def keep_fn(carry_inner):
                s, _ = carry_inner
                return s

            state = jax.lax.cond(
                done_prev,
                reset_fn,
                keep_fn,
                operand=(state, key_reset),
            )

            # Side-to-move observation
            obs = state.observation               # (5,5,115)
            time_left = state.time_left           # (2,)
            cur = state.current_player            # scalar 0/1

            my_time = time_left[cur]
            opp_time = time_left[1 - cur]
            time_norm = jnp.array(
                [my_time / default_time_f32, opp_time / default_time_f32],
                dtype=jnp.float32,
            )                                     # (2,)

            # -------- Gate decision --------
            obs_b = obs[None, ...]
            time_b = time_norm[None, ...]
            logits, value = gate_forward.apply(gate_params, obs_b, time_b)
            logits = logits[0]                    # (2,)
            value = value[0]                      # scalar

            log_probs = jax.nn.log_softmax(logits)
            gate_action = jax.random.categorical(key_gate, logits)  # 0 or 1
            logp = log_probs[gate_action]

            # -------- Execute move via chosen MCTS --------
            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)

            def run_mcts_8(_):
                a = select_mcts_8(model_8, state_b, key_mcts)[0]
                return a, jnp.int32(COST_8)

            def run_mcts_32(_):
                a = select_mcts_32(model_32, state_b, key_mcts)[0]
                return a, jnp.int32(COST_32)

            action, time_spent = jax.lax.cond(
                gate_action == 0,
                run_mcts_8,
                run_mcts_32,
                operand=None,
            )

            prev_player = cur
            state_next = env_speed.step(state, (action, time_spent))

            # Reward from POV of the player who just moved; 0 if non-terminal
            done = state_next.terminated | state_next.truncated
            move_reward = state_next.rewards[prev_player]
            reward = jax.lax.select(done, move_reward, jnp.float32(0.0))

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
            }
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            one_step, (state, rng_key), xs=None, length=num_steps
        )
        return state_final, rng_final, traj

    rollout_core_jit = jax.jit(rollout_core, static_argnums=(3,))
    return rollout_core_jit


def make_rollout_core_random(
    env_speed: GardnerChess,
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
):
    """
    Same as rollout_core, but gating policy is uniform random over {8,32}.
    """
    default_time_f32 = jnp.float32(default_time)

    def rollout_core_random(state, gate_params_unused, rng_key, num_steps: int):
        def one_step(carry, _):
            state, rng = carry

            rng, key_reset, key_gate, key_mcts = jax.random.split(rng, 4)
            done_prev = state.terminated | state.truncated

            def reset_fn(carry_inner):
                _state, k = carry_inner
                return env_speed.init(k)

            def keep_fn(carry_inner):
                s, _ = carry_inner
                return s

            state = jax.lax.cond(
                done_prev,
                reset_fn,
                keep_fn,
                operand=(state, key_reset),
            )

            obs = state.observation
            time_left = state.time_left
            cur = state.current_player

            my_time = time_left[cur]
            opp_time = time_left[1 - cur]
            time_norm = jnp.array(
                [my_time / default_time_f32, opp_time / default_time_f32],
                dtype=jnp.float32,
            )

            # Random gate: uniform {0,1}
            gate_action = jax.random.randint(key_gate, (), 0, 2)
            logp = jnp.log(jnp.float32(0.5))
            value = jnp.float32(0.0)  # unused in eval

            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)

            def run_mcts_8(_):
                a = select_mcts_8(model_8, state_b, key_mcts)[0]
                return a, jnp.int32(COST_8)

            def run_mcts_32(_):
                a = select_mcts_32(model_32, state_b, key_mcts)[0]
                return a, jnp.int32(COST_32)

            action, time_spent = jax.lax.cond(
                gate_action == 0,
                run_mcts_8,
                run_mcts_32,
                operand=None,
            )

            prev_player = cur
            state_next = env_speed.step(state, (action, time_spent))
            done = state_next.terminated | state_next.truncated
            move_reward = state_next.rewards[prev_player]
            reward = jax.lax.select(done, move_reward, jnp.float32(0.0))

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
            }
            return carry_next, out

        (state_final, rng_final), traj = jax.lax.scan(
            one_step, (state, rng_key), xs=None, length=num_steps
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
    default_time: float,
):
    """
    Wrapper: calls jitted rollout_core, then builds PPOBatch + stats + episode traces + hist.
    """
    state_final, rng_final, traj = rollout_core_jit(
        state, gate_params, rng_key, num_steps
    )

    batch, stats, ep_traces, hist_vals_8, hist_vals_32 = build_batch_and_stats(
        traj, default_time
    )
    return state_final, rng_final, batch, stats, ep_traces, hist_vals_8, hist_vals_32


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
    rng, key_init = jax.random.split(rng)
    state = env_speed.init(key_init)

    # Use initial clock as "default_time" for normalization
    default_time = float(state.time_left[0])

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
        project="pgx-speed-gate",
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
            "cost_8": COST_8,
            "cost_32": COST_32,
        },
        name="gate_speed_gardner_selfplay",
    )
    print("initialized wandb and checkpoint loading")

    run = wandb.run
    run_name = run.name if run is not None else "gate_run"
    save_dir = os.path.join(GATE_CKPT_ROOT, run_name)
    os.makedirs(save_dir, exist_ok=True)
    SAVE_INTERVAL = max(1, NUM_UPDATES // 10)
    best_mean_return = -1e9

    # ----------------------
    # Training loop
    # ----------------------
    for update in trange(1, NUM_UPDATES + 1, desc="PPO updates"):
        # Collect rollout (jitted)
        state, rng, batch, roll_stats, ep_traces, hist_vals_8, hist_vals_32 = rollout_selfplay(
            rollout_core_jit,
            state,
            gate_params,
            rng,
            ROLLOUT_STEPS,
            default_time,
        )

        # Shuffle batch and do minibatch PPO via single scan
        T = int(batch.obs.shape[0])
        num_mbs = T // BATCH_SIZE
        if num_mbs == 0:
            raise RuntimeError(f"Batch too small for BATCH_SIZE={BATCH_SIZE}, T={T}")

        all_mb_indices = []
        for epoch in range(PPO_EPOCHS):
            idxs = np.arange(T)
            np.random.shuffle(idxs)
            idxs_epoch = idxs[: num_mbs * BATCH_SIZE].reshape(num_mbs, BATCH_SIZE)
            all_mb_indices.append(idxs_epoch)
        all_mb_indices = np.stack(all_mb_indices, axis=0)  # (E, num_mbs, B)
        all_mb_indices = all_mb_indices.reshape(-1, BATCH_SIZE)  # (U,B)
        U = all_mb_indices.shape[0]

        mb_obs = batch.obs[all_mb_indices]            # (U,B,...)
        mb_time = batch.time[all_mb_indices]
        mb_actions = batch.actions[all_mb_indices]
        mb_logp = batch.logp_old[all_mb_indices]
        mb_values = batch.values_old[all_mb_indices]
        mb_returns = batch.returns[all_mb_indices]
        mb_advantages = batch.advantages[all_mb_indices]

        batches_seq = PPOBatch(
            obs=mb_obs,
            time=mb_time,
            actions=mb_actions,
            logp_old=mb_logp,
            values_old=mb_values,
            returns=mb_returns,
            advantages=mb_advantages,
        )

        gate_params, opt_state, metrics_seq = ppo_update_many(
            gate_params, opt_state, batches_seq
        )

        metrics_avg = {k: float(jnp.mean(v)) for k, v in metrics_seq.items()}

        # --------------------------
        # Random baseline & checkpointing every 10%
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
            state_eval = env_speed.init(key_eval_init)
            state_eval, rng, traj_random = rollout_core_random_jit(
                state_eval, gate_params, rng, ROLLOUT_STEPS
            )
            _, random_stats, _, _, _ = build_batch_and_stats(traj_random, default_time)

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
            "rollout/avg_final_time_p0": roll_stats["avg_final_time_p0"],
            "rollout/avg_final_time_p1": roll_stats["avg_final_time_p1"],
            "rollout/avg_final_time_p0_norm": roll_stats["avg_final_time_p0_norm"],
            "rollout/avg_final_time_p1_norm": roll_stats["avg_final_time_p1_norm"],
        }

        # Histograms: move index where nsim=8 / nsim=32 were used (across all games)
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

        # Sample *single* game: move index -> nsim & time
        if (update % TRAJ_LOG_INTERVAL == 0) and len(ep_traces) > 0:
            sample_ep = ep_traces[0]
            steps = sample_ep["steps"]
            L = len(steps)
            moves_idx = list(range(L))
            sims = [8 if s["action"] == 0 else 32 for s in steps]
            my_time_ticks = [s["my_time"] * default_time for s in steps]
            opp_time_ticks = [s["opp_time"] * default_time for s in steps]

            lines = [
                "Result P0={:.1f}, P1={:.1f}".format(
                    sample_ep["result_p0"], sample_ep["result_p1"]
                )
            ]
            for i, s in enumerate(steps):
                lines.append(
                    f"{i:02d} | P{s['player']} | "
                    f"nsim={8 if s['action'] == 0 else 32} | "
                    f"my_t={s['my_time']:.2f} opp_t={s['opp_time']:.2f}"
                )
            traj_text = "\n".join(lines)

            log_dict["debug/sample_game"] = traj_text
            log_dict["debug/sample_moves_idx"] = moves_idx
            log_dict["debug/sample_sims"] = sims
            log_dict["debug/sample_my_time_ticks"] = my_time_ticks
            log_dict["debug/sample_opp_time_ticks"] = opp_time_ticks
            log_dict["debug/sample_num_moves"] = L

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
