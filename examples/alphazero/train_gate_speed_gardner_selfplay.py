import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
from pydantic import BaseModel

from speed_gardner_chess import GardnerChess, MAX_TERMINATION_STEPS
from speed_gardner_chess import _step_board  # pure board step for MCTS
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
    return jax.jit(select_actions_mcts)


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

@dataclass
class PPOBatch:
    obs: jnp.ndarray         # (T, 5,5,115)
    time: jnp.ndarray        # (T, 2) normalized times
    actions: jnp.ndarray     # (T,)
    logp_old: jnp.ndarray    # (T,)
    values_old: jnp.ndarray  # (T,)
    returns: jnp.ndarray     # (T,)
    advantages: jnp.ndarray  # (T,)


def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones: (T,)
    Returns: returns, advantages
    """
    T = rewards.shape[0]
    returns = jnp.zeros_like(rewards)
    advantages = jnp.zeros_like(rewards)

    next_value = 0.0
    next_adv = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        adv = delta + gamma * lam * mask * next_adv
        advantages = advantages.at[t].set(adv)
        next_adv = adv
        next_value = values[t]
        returns = returns.at[t].set(adv + values[t])

    return returns, advantages


def make_ppo_loss_fn():
    def loss_fn(params, batch: PPOBatch):
        logits, values = gate_forward.apply(
            params, None, batch.obs, batch.time
        )  # logits: (T,2), values: (T,)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        logp = log_probs[jnp.arange(batch.actions.shape[0]), batch.actions]  # (T,)

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
# 6. Rollout: self-play, GateNet controls both players
# ================================================================

def rollout_selfplay(
    env: GardnerChess,
    state,
    gate_params,
    rng_key,
    select_mcts_8,
    select_mcts_32,
    model_8,
    model_32,
    default_time: float,
    num_steps: int,
):
    """
    Self-play rollout:
      - GateNet controls *both* players.
      - At each non-terminal state:
          * observation = state.observation (side-to-move)
          * times = [my_time, opp_time] / default_time
          * gate_action ∈ {0,1} selects nsim ∈ {8,32}
          * base MCTS picks move, env.step((action, time_spent))
          * reward r_t = final game result from the perspective
            of the player who just moved (mostly 0; ±1 or 0 at terminal)
    """
    obs_list = []
    time_list = []
    act_list = []
    logp_list = []
    val_list = []
    rew_list = []
    done_list = []

    ep_returns = []
    ep_lens = []

    ep_ret = 0.0
    ep_len = 0

    T_collected = 0
    rng = rng_key

    def gate_step(params, obs, time_norm, rng):
        # obs: (5,5,115), time_norm: (2,)
        obs_b = obs[None, ...]
        time_b = time_norm[None, ...]
        logits, value = gate_forward.apply(params, None, obs_b, time_b)
        logits = logits[0]
        value = value[0]
        log_probs = jax.nn.log_softmax(logits)
        rng, sub = jax.random.split(rng)
        action = int(jax.random.categorical(sub, logits))
        logp = float(log_probs[action])
        return action, float(value), logp, rng

    while T_collected < num_steps:
        if bool(state.terminated | state.truncated):
            # episode ended; log and reset
            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)
            ep_ret = 0.0
            ep_len = 0
            rng, key_init = jax.random.split(rng)
            state = env.init(key_init)
            continue

        # Side-to-move observation
        obs = state.observation           # (5,5,115)
        time_left = state.time_left       # (2,)
        cur = int(state.current_player)
        my_time = time_left[cur]
        opp_time = time_left[1 - cur]
        time_norm = jnp.array(
            [my_time / default_time, opp_time / default_time],
            dtype=jnp.float32,
        )

        # -------- Gate decision --------
        gate_action, value, logp, rng = gate_step(
            gate_params, obs, time_norm, rng
        )

        # Record transition (reward, done unknown yet)
        obs_list.append(np.array(obs))
        time_list.append(np.array(time_norm))
        act_list.append(int(gate_action))
        logp_list.append(float(logp))
        val_list.append(float(value))
        rew_list.append(0.0)      # will set to final result for this player at terminal
        done_list.append(False)
        ep_len += 1
        T_collected += 1

        # -------- Execute move via chosen MCTS --------
        rng, sub = jax.random.split(rng)
        state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)

        if gate_action == 0:
            a = select_mcts_8(model_8, state_b, sub)[0]
            time_spent = jnp.int32(COST_8)
        else:
            a = select_mcts_32(model_32, state_b, sub)[0]
            time_spent = jnp.int32(COST_32)

        prev_player = cur
        state = env.step(state, (a, time_spent))

        # Reward is from the perspective of prev_player (the one that decided)
        if bool(state.terminated | state.truncated):
            r = float(state.rewards[prev_player])
            rew_list[-1] = r
            done_list[-1] = True
            ep_ret += r
        else:
            # intermediate step, reward 0
            pass

    # ----------------------------
    # Build PPOBatch
    # ----------------------------
    obs_arr = jnp.array(obs_list, dtype=jnp.float32)            # (T,5,5,115)
    time_arr = jnp.array(time_list, dtype=jnp.float32)          # (T,2)
    actions_arr = jnp.array(act_list, dtype=jnp.int32)          # (T,)
    logp_arr = jnp.array(logp_list, dtype=jnp.float32)          # (T,)
    values_arr = jnp.array(val_list, dtype=jnp.float32)         # (T,)
    rewards_arr = jnp.array(rew_list, dtype=jnp.float32)        # (T,)
    dones_arr = jnp.array(done_list, dtype=jnp.float32)         # (T,)

    returns, advantages = compute_gae(
        rewards_arr, values_arr, dones_arr, GAMMA, LAMBDA
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

    ep_returns_np = np.array(ep_returns) if ep_returns else np.array([0.0])
    ep_lens_np = np.array(ep_lens) if ep_lens else np.array([0])

    stats = {
        "mean_ep_return": float(ep_returns_np.mean()),
        "mean_ep_len": float(ep_lens_np.mean()),
    }

    return state, rng, batch, stats


# ================================================================
# 7. Main training loop (self-play PPO)
# ================================================================

def main():
    # ----------------------
    # Load base models
    # ----------------------
    ckpt_paths = discover_checkpoints(CKPT_ROOT, BASE_ENV_ID, ITER_FILE)

    # Must have nsim_8 and nsim_32
    needed = [f"nsim_{n}" for n in BASE_NSIMS]
    for k in needed:
        if k not in ckpt_paths:
            raise RuntimeError(f"Missing checkpoint for {k} in {CKPT_ROOT}")

    env_id_8, cfg_8, model_8 = load_checkpoint(ckpt_paths["nsim_8"])
    env_id_32, cfg_32, model_32 = load_checkpoint(ckpt_paths["nsim_32"])

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

    forward = build_forward(env_speed, cfg_8)  # same arch for both

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
    gate_params = gate_forward.init(jax.random.PRNGKey(SEED + 123), dummy_obs, dummy_time)

    optimizer = optax.adam(PPO_LR)
    opt_state = optimizer.init(gate_params)

    loss_fn = make_ppo_loss_fn()
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def ppo_update_step(params, opt_state, batch: PPOBatch):
        (loss, metrics), grads = value_and_grad(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    # ----------------------
    # Training loop
    # ----------------------
    for update in range(1, NUM_UPDATES + 1):
        # Collect rollout
        state, rng, batch, roll_stats = rollout_selfplay(
            env_speed,
            state,
            gate_params,
            rng,
            select_mcts_8,
            select_mcts_32,
            model_8,
            model_32,
            default_time,
            ROLLOUT_STEPS,
        )

        # Shuffle batch and do minibatch PPO
        T = batch.obs.shape[0]
        idxs = np.arange(T)
        np.random.shuffle(idxs)

        # We'll accumulate metrics average over epochs
        metrics_accum = None
        for epoch in range(PPO_EPOCHS):
            for start in range(0, T, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = idxs[start:end]
                if len(mb_idx) == 0:
                    continue

                mb = PPOBatch(
                    obs=batch.obs[mb_idx],
                    time=batch.time[mb_idx],
                    actions=batch.actions[mb_idx],
                    logp_old=batch.logp_old[mb_idx],
                    values_old=batch.values_old[mb_idx],
                    returns=batch.returns[mb_idx],
                    advantages=batch.advantages[mb_idx],
                )
                gate_params, opt_state, metrics = ppo_update_step(
                    gate_params, opt_state, mb
                )
                if metrics_accum is None:
                    metrics_accum = {k: float(v) for k, v in metrics.items()}
                else:
                    for k, v in metrics.items():
                        metrics_accum[k] += float(v)

        if metrics_accum is not None:
            # normalize metric by number of minibatches * epochs
            n_mbs = PPO_EPOCHS * max(1, T // BATCH_SIZE)
            metrics_avg = {k: v / n_mbs for k, v in metrics_accum.items()}
        else:
            metrics_avg = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                           "entropy": 0.0, "approx_kl": 0.0}

        if update % 10 == 0:
            print(
                f"[Update {update}] "
                f"loss={metrics_avg['loss']:.4f}  "
                f"policy={metrics_avg['policy_loss']:.4f}  "
                f"value={metrics_avg['value_loss']:.4f}  "
                f"entropy={metrics_avg['entropy']:.4f}  "
                f"KL={metrics_avg['approx_kl']:.6f}  "
                f"mean_ep_return={roll_stats['mean_ep_return']:.3f}  "
                f"mean_ep_len={roll_stats['mean_ep_len']:.1f}"
            )


if __name__ == "__main__":
    main()
