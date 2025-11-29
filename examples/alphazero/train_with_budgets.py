# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network_budgets import AZNet
from speed_gardner_chess import GardnerChess, MAX_TERMINATION_STEPS  # speed Gardner env

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    # Note: env_id is used only for baseline + logging; env itself is constructed manually
    env_id: pgx.EnvId = "gardner_chess"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    # We treat num_simulations as the "high" budget, e.g. 32
    num_simulations: int = 32
    # Low budget (e.g. 16; set to 8 to test 8 vs 32)
    sim_budget_low: int = 4
    max_num_steps: int = 256
    # time-control params (in "ticks")
    # One simulation costs sim_tick_cost ticks of clock time.
    sim_tick_cost: int = 5
    # Used to set initial time per player:
    #   initial_time = num_simulations (high) * sim_tick_cost * avg_moves
    avg_moves: int = 3
    # Baseline pays a very small fixed cost per move so it basically never flags.
    baseline_tick_per_move: int = 1
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # coefficient on the simulation-head REINFORCE loss
    sim_head_coef: float = 0.1
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

# ----------------------------------------------------------------------
# Environment and time budget
# ----------------------------------------------------------------------
env = GardnerChess()

# Pre-trained reference opponent from PGX (original GardnerChess, not speed version)
baseline = pgx.make_baseline_model(config.env_id + "_v0")

SIM_BUDGET_HIGH = int(config.num_simulations)
SIM_BUDGET_LOW = int(config.sim_budget_low)

# Per-game initial time budget (per player) and per-move cost for our AZ agent
INITIAL_TIME = SIM_BUDGET_HIGH * config.sim_tick_cost * config.avg_moves  # ticks
MOVE_TICK_COST_HIGH = SIM_BUDGET_HIGH * config.sim_tick_cost
MOVE_TICK_COST_LOW = SIM_BUDGET_LOW * config.sim_tick_cost
BASELINE_TICK_COST = config.baseline_tick_per_move  # baseline per move


def init_with_custom_time(key: jnp.ndarray):
    """Wrapper around env.init that sets a custom per-player time budget."""
    state = env.init(key)
    initial = jnp.int32(INITIAL_TIME)
    return state.replace(_time_left=jnp.int32([initial, initial]))


# For self-play we want automatic resetting when an episode terminates
step_with_auto_reset = auto_reset(env.step, init_with_custom_time)


# ----------------------------------------------------------------------
# Network + optimizer
# ----------------------------------------------------------------------
def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out, sim_logits = net(
        x, is_training=not is_eval, test_local_stats=False
    )
    return policy_out, value_out, sim_logits


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


# ----------------------------------------------------------------------
# MuZero recurrent function (no clock cost inside simulations!)
# ----------------------------------------------------------------------
def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: (params, state)
    del rng_key
    model_params, model_state = model

    current_player = state.current_player

    # IMPORTANT: here we call env.step with only the action,
    # so that MCTS simulations do NOT consume clock time.
    state = jax.vmap(env.step)(state, action)

    (logits, value, _), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(
        state.legal_action_mask, logits, jnp.finfo(logits.dtype).min
    )

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


# ----------------------------------------------------------------------
# Self-play
# ----------------------------------------------------------------------
class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    time_left: jnp.ndarray      # (T, B, 2)
    step_count: jnp.ndarray     # (T, B)
    budget_id: jnp.ndarray      # (T, B)  0=low, 1=high


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray):
    """Run batched self-play with adaptive simulation budgets.

    For each step:
      - run MCTS with low and high budgets
      - simulation head chooses low/high per env
      - pick corresponding action, action_weights, and time cost
    Returns:
      data: SelfplayOutput  (T, B, ...)
      final_state: State    (#envs on this device)
    """
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    low_ticks = jnp.int32(MOVE_TICK_COST_LOW)
    high_ticks = jnp.int32(MOVE_TICK_COST_HIGH)

    def step_fn(state, key) -> tuple[pgx.State, SelfplayOutput]:
        key_trunk, key_mcts, key_reset = jax.random.split(key, 3)
        observation = state.observation

        (logits, value, sim_logits), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        # Run low- and high-budget MCTS for the entire batch
        key_lo, key_hi, key_budget = jax.random.split(key_mcts, 3)
        policy_output_low = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key_lo,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=SIM_BUDGET_LOW,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        policy_output_high = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key_hi,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=SIM_BUDGET_HIGH,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        # Simulation-budget head chooses low vs high per env
        # sim_logits: (B, 2)
        budget_id = jax.random.categorical(
            key_budget, sim_logits, axis=-1
        )  # (B,), values in {0,1}

        actor = state.current_player
        keys = jax.random.split(key_reset, batch_size)

        # Select actions and action_weights according to budget_id
        low_action = policy_output_low.action
        high_action = policy_output_high.action
        action = jnp.where(budget_id == 0, low_action, high_action)

        low_aw = policy_output_low.action_weights
        high_aw = policy_output_high.action_weights
        budget_id_exp = budget_id[:, None]
        action_weights = jnp.where(budget_id_exp == 0, low_aw, high_aw)

        # Time cost depends on chosen budget
        time_spent = jnp.where(budget_id == 0, low_ticks, high_ticks).astype(
            jnp.int32
        )

        # Auto-reset terminated environments, and step with (action, time_spent)
        state = jax.vmap(step_with_auto_reset)(
            state, (action, time_spent), keys
        )

        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
            time_left=state.time_left,
            step_count=state._step_count,
            budget_id=budget_id,
        )

    # Run selfplay for max_num_steps
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(init_with_custom_time)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    final_state, data = jax.lax.scan(step_fn, state, key_seq)

    return data, final_state


# ----------------------------------------------------------------------
# Loss + training
# ----------------------------------------------------------------------
class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray
    budget_id: jnp.ndarray  # (T, B)


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target (backward discounted return)
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
        budget_id=data.budget_id,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value, sim_logits), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    # -------------------------
    # Standard AlphaZero losses
    # -------------------------
    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(
        value_loss * samples.mask
    )  # mask if the episode is truncated

    # -------------------------
    # Simulation-budget REINFORCE loss
    # -------------------------
    # sim_logits: (N, 2)
    # budget_id:  (N,)
    budget_one_hot = jax.nn.one_hot(samples.budget_id, num_classes=2)
    log_probs = jax.nn.log_softmax(sim_logits, axis=-1)
    chosen_log_prob = jnp.sum(log_probs * budget_one_hot, axis=-1)  # (N,)

    # Advantage: (R - V_baseline), masked on valid value targets
    advantage = samples.value_tgt - jax.lax.stop_gradient(value)
    advantage = advantage * samples.mask
    sim_loss = -jnp.mean(chosen_log_prob * advantage)

    total_loss = policy_loss + value_loss + config.sim_head_coef * sim_loss
    return total_loss, (model_state, policy_loss, value_loss, sim_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss, sim_loss) = jax.grad(
        loss_fn, has_aux=True
    )(model_params, model_state, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss, sim_loss


# ----------------------------------------------------------------------
# Evaluation vs baseline (with time-cost)
# ----------------------------------------------------------------------
@jax.pmap
def evaluate(rng_key, my_model):
    """A simplified evaluation by sampling.

    Here we keep evaluation simple: our agent uses policy logits directly
    (no MCTS) but still pays a fixed time per move corresponding to the
    *high* compute budget. Baseline pays a tiny fixed cost.
    """
    my_player = 0
    my_model_params, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(init_with_custom_time)(keys)

    move_ticks = jnp.int32(MOVE_TICK_COST_HIGH)
    baseline_ticks = jnp.int32(BASELINE_TICK_COST)

    def body_fn(val):
        key, state, R = val
        (my_logits, _, _), _ = forward.apply(
            my_model_params, my_model_state, state.observation, is_eval=True
        )
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)

        # Time-cost per move depends on whose turn it is
        is_my_turn_flat = (state.current_player == my_player)
        time_spent = jnp.where(is_my_turn_flat, move_ticks, baseline_ticks)

        state = jax.vmap(env.step)(state, (action, time_spent))
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (key, state, jnp.zeros(batch_size)),
    )
    return R


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(project="pgx-az", config=config.model_dump(), mode="online", name=f"B={config.avg_moves}_S={config.sim_budget_low}_H={config.num_simulations}")
    print("JAX devices:", jax.local_devices())
    print("num_devices:", num_devices)

    # Initialize model and opt_state
    dummy_state = jax.vmap(init_with_custom_time)(
        jax.random.split(jax.random.PRNGKey(0), 2)
    )
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])
    # replicate to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if iteration % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, model)
            log.update(
                {
                    "eval/vs_baseline/avg_R": R.mean().item(),
                    "eval/vs_baseline/win_rate": (
                        (R == 1).sum() / R.size
                    ).item(),
                    "eval/vs_baseline/draw_rate": (
                        (R == 0).sum() / R.size
                    ).item(),
                    "eval/vs_baseline/lose_rate": (
                        (R == -1).sum() / R.size
                    ).item(),
                }
            )

            # Store checkpoints
            model_0, opt_state_0 = jax.tree_util.tree_map(
                lambda x: x[0], (model, opt_state)
            )
            with open(
                os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb"
            ) as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # --------------------------------------------------------------
        # Selfplay
        # --------------------------------------------------------------
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data, final_state = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Move data to host for stats
        data_host: SelfplayOutput = jax.device_get(data)

        # Shapes:
        # data_host.terminated: (num_devices, T, B)
        # data_host.reward:     (num_devices, T, B)
        # data_host.time_left:  (num_devices, T, B, 2)
        # data_host.step_count: (num_devices, T, B)
        # data_host.budget_id:  (num_devices, T, B)
        T = config.max_num_steps
        last_t = T - 1

        terminated = data_host.terminated[:, last_t, :]    # (D, B)
        reward_last = data_host.reward[:, last_t, :]       # (D, B)
        time_left_last = data_host.time_left[:, last_t, :, :]  # (D, B, 2)
        step_count_last = data_host.step_count[:, last_t, :]    # (D, B)
        budget_last = data_host.budget_id[:, last_t, :]    # (D, B)

        # Flatten device + batch
        term_flat = terminated.reshape(-1).astype(jnp.bool_)
        reward_flat = reward_last.reshape(-1)
        time_left_flat = time_left_last.reshape(-1, 2)
        step_count_flat = step_count_last.reshape(-1)
        budget_flat = budget_last.reshape(-1)

        total_envs = term_flat.shape[0]

        # Overall termination / truncation at last step
        term_rate = float(jnp.mean(term_flat.astype(jnp.float32)))
        trunc_rate = 1.0 - term_rate

        # Timeouts: terminated and some player's clock <= 0
        timed_out_mask = term_flat & (jnp.any(time_left_flat <= 0, axis=-1))

        # Env-level max-step terminations (Gardner's MAX_TERMINATION_STEPS)
        env_max_mask = term_flat & (step_count_flat >= MAX_TERMINATION_STEPS)

        # Board results (excluding timeouts + env-max)
        non_timeout_non_max = term_flat & ~timed_out_mask & ~env_max_mask
        win_mask = non_timeout_non_max & (reward_flat > 0.0)
        loss_mask = non_timeout_non_max & (reward_flat < 0.0)
        draw_mask = non_timeout_non_max & jnp.isclose(reward_flat, 0.0)

        def frac(mask):
            return float(
                jnp.sum(mask.astype(jnp.float32)) / float(total_envs)
            )

        timeout_rate = frac(timed_out_mask)
        env_max_rate = frac(env_max_mask)
        board_win_rate = frac(win_mask)
        board_loss_rate = frac(loss_mask)
        board_draw_rate = frac(draw_mask)

        # How often did we choose the high budget?
        budget_high_rate = float(
            jnp.mean((budget_flat == 1).astype(jnp.float32))
        )

        # Extra self-play stats (clock + move count-ish on final_state)
        avg_step_count = float(jnp.mean(final_state._step_count))
        avg_time_left_p0 = float(jnp.mean(final_state.time_left[..., 0]))
        avg_time_left_p1 = float(jnp.mean(final_state.time_left[..., 1]))

        # --------------------------------------------------------------
        # Shuffle samples and make minibatches
        # --------------------------------------------------------------
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += (
            samples.obs.shape[0]
            * samples.obs.shape[1]
            * samples.obs.shape[2]
        )
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])), samples
        )
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]),
            samples,
        )

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        policy_losses, value_losses, sim_losses = [], [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(
                lambda x: x[i], minibatches
            )
            model, opt_state, policy_loss, value_loss, sim_loss = train(
                model, opt_state, minibatch
            )
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
            sim_losses.append(sim_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)
        sim_loss = sum(sim_losses) / len(sim_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "train/sim_head_loss": sim_loss,
                "hours": hours,
                "frames": frames,
                # Extra speed-chess stats
                "selfplay/avg_step_count": avg_step_count,
                "selfplay/avg_time_left_p0": avg_time_left_p0,
                "selfplay/avg_time_left_p1": avg_time_left_p1,
                "selfplay/termination_rate_last": term_rate,
                "selfplay/truncation_rate_last": trunc_rate,
                "selfplay/term_rate_timeout": timeout_rate,
                "selfplay/term_rate_env_max_steps": env_max_rate,
                "selfplay/term_rate_board_win": board_win_rate,
                "selfplay/term_rate_board_loss": board_loss_rate,
                "selfplay/term_rate_board_draw": board_draw_rate,
                "selfplay/budget_high_rate": budget_high_rate,
            }
        )
