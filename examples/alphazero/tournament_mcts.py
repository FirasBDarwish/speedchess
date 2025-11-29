from __future__ import annotations

import dataclasses
import os
import pickle
import time
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import pgx
from pydantic import BaseModel
import wandb

from network import AZNet  # same AZNet you used for training


# ---------------------------------------------------------------------
# Config stub for unpickling (matches training script)
# ---------------------------------------------------------------------
class Config(BaseModel):
    env_id: pgx.EnvId = "gardner_chess"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


# Map any pickled "Config" class to this one
class ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Config":
            return Config
        return super().find_class(module, name)


def load_checkpoint(path: str):
    """Load a checkpoint and return (env_id, cfg, model)."""
    with open(path, "rb") as f:
        data = ConfigUnpickler(f).load()
    model = data["model"]      # (params, state)
    cfg: Config = data["config"]
    env_id = data.get("env_id", cfg.env_id)
    return env_id, cfg, model


def build_forward(env: pgx.Env, cfg: Config):
    """Rebuild the same Haiku net as in training."""
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


def make_recurrent_fn(env: pgx.Env, forward):
    """
    MuZero recurrent function, same structure as in your training script.
    """
    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        # model: (params, model_state)
        del rng_key
        model_params, model_state = model

        current_player = state.current_player
        # step env in batch
        state = jax.vmap(env.step)(state, action)

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        # rewards from POV of player who just moved
        batch_size = state.rewards.shape[0]
        reward = state.rewards[jnp.arange(batch_size), current_player]

        # zero-out beyond terminal
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


# ---------------------------------------------------------------------
# Stats structures
# ---------------------------------------------------------------------
@dataclasses.dataclass
class ModelStats:
    name: str
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    steps_win_sum: float = 0.0
    steps_loss_sum: float = 0.0
    steps_draw_sum: float = 0.0

    def describe(self):
        def ratio(n):
            return n / self.total_games if self.total_games else 0.0

        def avg(total, n):
            return total / n if n > 0 else float("nan")

        return {
            "games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": ratio(self.wins),
            "loss_rate": ratio(self.losses),
            "draw_rate": ratio(self.draws),
            "avg_moves_win": avg(self.steps_win_sum, self.wins),
            "avg_moves_loss": avg(self.steps_loss_sum, self.losses),
            "avg_moves_draw": avg(self.steps_draw_sum, self.draws),
        }


def discover_checkpoints(
    root: str,
    env_id: str,
    iter_filename: str = "000400.ckpt",
) -> Dict[str, str]:
    """
    Find checkpoints like:
      root/nsim_2/gardner_chess_YYYY.../000400.ckpt
      root/nsim_4/gardner_chess_YYYY.../000400.ckpt
    Returns a dict { "nsim_2": path, ... }.
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


# ---------------------------------------------------------------------
# Main tournament, with per-model MCTS budget == training num_simulations
# ---------------------------------------------------------------------
def main():
    # --- tweak these as needed ---
    CKPT_ROOT = "/n/home04/amuppidi/speedchess/examples/alphazero/checkpoints"
    ENV_ID = "gardner_chess"
    ITER_FILE = "000400.ckpt"          # iteration to compare
    NUM_GAMES_PER_COLOR = 64           # games as White + games as Black per pairing
    SEED = 0
    WANDB_PROJECT = "pgx-az-tournament"
    # For timing micro-benchmark
    TIMING_BATCH_SIZE = 64             # number of parallel boards for timing
    TIMING_REPS = 5                    # number of timed runs per model
    # ------------------------------

    ckpt_paths = discover_checkpoints(CKPT_ROOT, ENV_ID, ITER_FILE)
    if not ckpt_paths:
        raise RuntimeError("No checkpoints found; check CKPT_ROOT/ENV_ID/ITER_FILE")

    # Use first checkpoint to recover env + architecture
    first_name = sorted(ckpt_paths.keys())[0]
    env_id0, cfg0, model0 = load_checkpoint(ckpt_paths[first_name])
    if env_id0 != ENV_ID:
        raise RuntimeError(f"Checkpoint env_id {env_id0} != expected {ENV_ID}")

    env = pgx.make(env_id0)
    forward = build_forward(env, cfg0)
    recurrent_fn = make_recurrent_fn(env, forward)

    max_steps = int(cfg0.max_num_steps)
    num_players = env.num_players

    # Load all models + their configs
    models: Dict[str, Tuple[Tuple, Config]] = {first_name: (model0, cfg0)}
    for name, path in ckpt_paths.items():
        if name == first_name:
            continue
        env_id, cfg, model = load_checkpoint(path)
        if env_id != env_id0:
            raise RuntimeError(f"Env id mismatch for {name}: {env_id} != {env_id0}")
        if cfg.num_layers != cfg0.num_layers or cfg.num_channels != cfg0.num_channels:
            raise RuntimeError(
                f"Network arch mismatch for {name} "
                f"(layers {cfg.num_layers} vs {cfg0.num_layers}, "
                f"channels {cfg.num_channels} vs {cfg0.num_channels})"
            )
        if cfg.max_num_steps != cfg0.max_num_steps:
            raise RuntimeError(
                f"max_num_steps mismatch for {name} "
                f"({cfg.max_num_steps} vs {cfg0.max_num_steps})"
            )
        models[name] = (model, cfg)

    model_names = sorted(models.keys())
    print("Models:")
    model_cfg_summary = {}
    for name in model_names:
        _, cfg = models[name]
        print(f"  {name}: num_simulations={cfg.num_simulations}")
        model_cfg_summary[name] = {
            "num_simulations": int(cfg.num_simulations),
            "num_layers": int(cfg.num_layers),
            "num_channels": int(cfg.num_channels),
        }

    # Init W&B
    wandb_config = {
        "ckpt_root": CKPT_ROOT,
        "env_id": ENV_ID,
        "iter_file": ITER_FILE,
        "num_games_per_color": NUM_GAMES_PER_COLOR,
        "seed": SEED,
        "models": model_cfg_summary,
    }
    wandb.init(
        project=WANDB_PROJECT,
        config=wandb_config,
        name=f"tournament_{ENV_ID}_{ITER_FILE}",
    )

    stats: Dict[str, ModelStats] = {name: ModelStats(name) for name in model_names}
    pair_results: Dict[Tuple[str, str], Dict[str, int]] = {}

    rng = jax.random.PRNGKey(SEED)

    # --- MCTS move selection where model uses its own num_sims (int) ---
    def select_actions_mcts(model, num_sims: int, state: pgx.State, rng_key: jnp.ndarray):
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
            num_simulations=int(num_sims),
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action  # (batch,)

    def update_stats(name: str, outcome, steps):
        """
        outcome: +1 win, -1 loss, 0 draw (from that model's POV)
        steps:   number of moves (total step_count) for that game
        """
        s = stats[name]
        s.total_games += int(outcome.shape[0])

        win_mask = outcome > 0
        loss_mask = outcome < 0
        draw_mask = outcome == 0

        s.wins += int(win_mask.sum())
        s.losses += int(loss_mask.sum())
        s.draws += int(draw_mask.sum())

        s.steps_win_sum += float(steps[win_mask].sum())
        s.steps_loss_sum += float(steps[loss_mask].sum())
        s.steps_draw_sum += float(steps[draw_mask].sum())

    # small helper for per-pair move stats
    def compute_move_stats(outcomes, steps):
        import numpy as np

        outcomes = np.asarray(outcomes)
        steps = np.asarray(steps)

        mask_win = outcomes > 0
        mask_loss = outcomes < 0
        mask_draw = outcomes == 0

        def avg(msk):
            if not np.any(msk):
                return float("nan")
            return float(steps[msk].mean())

        return {
            "avg_moves_win": avg(mask_win),
            "avg_moves_loss": avg(mask_loss),
            "avg_moves_draw": avg(mask_draw),
        }

    # -----------------------------------------------------------------
    # Round-robin tournament
    # -----------------------------------------------------------------
    for i, na in enumerate(model_names):
        for nb in model_names[i + 1:]:
            (model_a, cfg_a) = models[na]
            (model_b, cfg_b) = models[nb]
            num_sims_a = int(cfg_a.num_simulations)
            num_sims_b = int(cfg_b.num_simulations)

            print(
                f"\n=== {na} (sims={num_sims_a}) "
                f"vs {nb} (sims={num_sims_b}) ==="
            )

            # ----- build a per-pair jitted match function -----
            def run_match(rng_key, model_a, model_b):
                keys = jax.random.split(rng_key, NUM_GAMES_PER_COLOR)
                state = jax.vmap(env.init)(keys)
                R = jnp.zeros((NUM_GAMES_PER_COLOR, num_players), dtype=jnp.float32)
                step = jnp.array(0, dtype=jnp.int32)

                def cond_fn(carry):
                    step, state, R, rng_key = carry
                    done_all = state.terminated.all()
                    return jnp.logical_and(step < max_steps, ~done_all)

                def body_fn(carry):
                    step, state, R, rng_key = carry
                    rng_key, key_a, key_b = jax.random.split(rng_key, 3)

                    # Each model runs MCTS with its own num_simulations
                    actions_a = select_actions_mcts(model_a, num_sims_a, state, key_a)
                    actions_b = select_actions_mcts(model_b, num_sims_b, state, key_b)

                    # Select whose action to use on each board
                    current_player = state.current_player
                    actions = jnp.where(current_player == 0, actions_a, actions_b)

                    # Step env and accumulate rewards
                    state = jax.vmap(env.step)(state, actions)
                    R = R + state.rewards

                    step = step + 1
                    return step, state, R, rng_key

                step, state, R, rng_key = jax.lax.while_loop(
                    cond_fn, body_fn, (step, state, R, rng_key)
                )
                return state, R

            run_match_jit = jax.jit(run_match)

            # Orientation 1: na as player 0, nb as player 1
            rng, key1 = jax.random.split(rng)
            final1, R1 = run_match_jit(key1, model_a, model_b)
            final1 = jax.device_get(final1)
            R1 = jax.device_get(R1)
            steps1 = final1._step_count  # (batch,)

            # Orientation 2: nb as player 0, na as player 1
            rng, key2 = jax.random.split(rng)

            def run_match_swap(rng_key, model_b, model_a):
                keys = jax.random.split(rng_key, NUM_GAMES_PER_COLOR)
                state = jax.vmap(env.init)(keys)
                R = jnp.zeros((NUM_GAMES_PER_COLOR, num_players), dtype=jnp.float32)
                step = jnp.array(0, dtype=jnp.int32)

                def cond_fn(carry):
                    step, state, R, rng_key = carry
                    done_all = state.terminated.all()
                    return jnp.logical_and(step < max_steps, ~done_all)

                def body_fn(carry):
                    step, state, R, rng_key = carry
                    rng_key, key_b, key_a = jax.random.split(rng_key, 3)

                    actions_b = select_actions_mcts(model_b, num_sims_b, state, key_b)
                    actions_a = select_actions_mcts(model_a, num_sims_a, state, key_a)

                    current_player = state.current_player
                    actions = jnp.where(current_player == 0, actions_b, actions_a)

                    state = jax.vmap(env.step)(state, actions)
                    R = R + state.rewards
                    step = step + 1
                    return step, state, R, rng_key

                step, state, R, rng_key = jax.lax.while_loop(
                    cond_fn, body_fn, (step, state, R, rng_key)
                )
                return state, R

            run_match_swap_jit = jax.jit(run_match_swap)

            final2, R2 = run_match_swap_jit(key2, model_b, model_a)
            final2 = jax.device_get(final2)
            R2 = jax.device_get(R2)
            steps2 = final2._step_count

            # Convert to numpy for aggregation
            import numpy as np
            R1 = np.asarray(R1)
            R2 = np.asarray(R2)
            steps1 = np.asarray(steps1)
            steps2 = np.asarray(steps2)

            # From R[:, 0] (player-0's return) build outcomes
            outcome_A1 = np.sign(R1[:, 0])  # na as player 0
            outcome_B1 = -outcome_A1        # nb as player 1

            outcome_B2 = np.sign(R2[:, 0])  # nb as player 0
            outcome_A2 = -outcome_B2        # na as player 1

            # Update per-model global stats
            update_stats(na, outcome_A1, steps1)
            update_stats(nb, outcome_B1, steps1)
            update_stats(nb, outcome_B2, steps2)
            update_stats(na, outcome_A2, steps2)

            # Pairwise tallies (global)
            wins_A = int((outcome_A1 > 0).sum() + (outcome_A2 > 0).sum())
            wins_B = int((outcome_B1 > 0).sum() + (outcome_B2 > 0).sum())
            draws = int((outcome_A1 == 0).sum() + (outcome_A2 == 0).sum())
            total_games = 2 * NUM_GAMES_PER_COLOR

            pair_results[(na, nb)] = {
                "games": total_games,
                f"{na}_wins": wins_A,
                f"{nb}_wins": wins_B,
                "draws": draws,
            }
            print(
                f"{na} wins: {wins_A}, {nb} wins: {wins_B}, "
                f"draws: {draws} (out of {total_games})"
            )

            # -----------------------------------------------------------------
            # Per-opponent move stats for wandb (pair-level)
            # -----------------------------------------------------------------
            # Merge orientations for per-opponent stats
            na_outcomes = np.concatenate([outcome_A1, outcome_A2], axis=0)
            nb_outcomes = np.concatenate([outcome_B1, outcome_B2], axis=0)
            # steps are symmetric for a given game
            pair_steps = np.concatenate([steps1, steps2], axis=0)

            na_move_stats = compute_move_stats(na_outcomes, pair_steps)
            nb_move_stats = compute_move_stats(nb_outcomes, pair_steps)

            pair_log = {
                f"pair/{na}_vs_{nb}/games": total_games,
                f"pair/{na}_vs_{nb}/{na}_wins": wins_A,
                f"pair/{na}_vs_{nb}/{nb}_wins": wins_B,
                f"pair/{na}_vs_{nb}/draws": draws,
                f"pair/{na}_vs_{nb}/{na}_win_rate": wins_A / total_games,
                f"pair/{na}_vs_{nb}/{nb}_win_rate": wins_B / total_games,
                f"pair/{na}_vs_{nb}/{na}_avg_moves_win": na_move_stats["avg_moves_win"],
                f"pair/{na}_vs_{nb}/{na}_avg_moves_loss": na_move_stats["avg_moves_loss"],
                f"pair/{na}_vs_{nb}/{na}_avg_moves_draw": na_move_stats["avg_moves_draw"],
                f"pair/{na}_vs_{nb}/{nb}_avg_moves_win": nb_move_stats["avg_moves_win"],
                f"pair/{na}_vs_{nb}/{nb}_avg_moves_loss": nb_move_stats["avg_moves_loss"],
                f"pair/{na}_vs_{nb}/{nb}_avg_moves_draw": nb_move_stats["avg_moves_draw"],
            }
            wandb.log(pair_log)

    # -----------------------------------------------------------------
    # Final per-pair printout
    # -----------------------------------------------------------------
    print("\n=== Per-pair results ===")
    for (na, nb), res in pair_results.items():
        print(f"{na} vs {nb}: {res}")

    # -----------------------------------------------------------------
    # Per-model global summary + wandb logging
    # -----------------------------------------------------------------
    print("\n=== Per-model summary ===")
    for name in model_names:
        d = stats[name].describe()
        print(f"{name}:")
        print(
            f"  games={d['games']}  W/L/D = "
            f"{d['wins']}/{d['losses']}/{d['draws']} "
            f"(win_rate={d['win_rate']:.3f})"
        )
        print(
            f"  avg_moves (win/loss/draw) = "
            f"{d['avg_moves_win']:.2f} / "
            f"{d['avg_moves_loss']:.2f} / "
            f"{d['avg_moves_draw']:.2f}"
        )

        model_log = {
            f"model/{name}/games": d["games"],
            f"model/{name}/wins": d["wins"],
            f"model/{name}/losses": d["losses"],
            f"model/{name}/draws": d["draws"],
            f"model/{name}/win_rate": d["win_rate"],
            f"model/{name}/loss_rate": d["loss_rate"],
            f"model/{name}/draw_rate": d["draw_rate"],
            f"model/{name}/avg_moves_win": d["avg_moves_win"],
            f"model/{name}/avg_moves_loss": d["avg_moves_loss"],
            f"model/{name}/avg_moves_draw": d["avg_moves_draw"],
        }
        wandb.log(model_log)

    # -----------------------------------------------------------------
    # Inference timing micro-benchmark (per model, per decision)
    # -----------------------------------------------------------------
    print("\n=== Inference timing (approx, per decision) ===")
    timing_results = {}

    for idx, name in enumerate(model_names):
        model, cfg = models[name]
        num_sims = int(cfg.num_simulations)

        # Use deterministic states for timing (all initial states here)
        key_states = jax.random.PRNGKey(SEED + 12345 + idx)
        keys = jax.random.split(key_states, TIMING_BATCH_SIZE)
        state = jax.vmap(env.init)(keys)

        def mcts_once(state, rng_key):
            return select_actions_mcts(model, num_sims, state, rng_key)

        mcts_once_jit = jax.jit(mcts_once)

        # Warmup (compilation)
        warm_key = jax.random.PRNGKey(SEED + 9999 + idx)
        _ = jax.block_until_ready(mcts_once_jit(state, warm_key))

        total_time = 0.0
        for rep in range(TIMING_REPS):
            rep_key = jax.random.PRNGKey(SEED + 100000 * (idx + 1) + rep)
            t0 = time.time()
            actions = mcts_once_jit(state, rep_key)
            _ = jax.block_until_ready(actions)
            t1 = time.time()
            total_time += (t1 - t0)

        avg_batch_time = total_time / TIMING_REPS
        avg_decision_time = avg_batch_time / float(TIMING_BATCH_SIZE)
        timing_results[name] = avg_decision_time

        print(f"{name} (sims={num_sims}): "
              f"{avg_batch_time:.4f}s per batch of {TIMING_BATCH_SIZE} "
              f"~ {avg_decision_time*1000:.3f} ms / decision")

        wandb.log({
            f"timing/{name}/avg_batch_time_s": avg_batch_time,
            f"timing/{name}/avg_decision_time_s": avg_decision_time,
            f"timing/{name}/num_simulations": num_sims,
        })


if __name__ == "__main__":
    main()
