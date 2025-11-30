import os
import io
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from pydantic import BaseModel

import svgwrite
import cairosvg
import imageio.v2 as imageio

from speed_gardner_chess import (
    GardnerChess,
    State,                # State type
    _step_board,
    _observe,
    DEFAULT_TIME,
    MAX_TERMINATION_STEPS,
)
from network import AZNet

# ================================================================
# Paths / constants
# ================================================================

CKPT_ROOT = "/n/home04/amuppidi/speedchess/examples/alphazero/checkpoints"
BASE_ENV_ID = "gardner_chess"
ITER_FILE = "000400.ckpt"
BASE_NSIMS = [8, 32]

GATE_CKPT_PATH = (
    "/n/home04/amuppidi/speedchess/examples/alphazero/"
    "gate_checkpoints_v/gate_update_001000.pkl"
)

COST_8 = 8
COST_32 = 32

GAME_GIF_DIR = "gate_eval_games_gif"
os.makedirs(GAME_GIF_DIR, exist_ok=True)
for sub in ("p0Win", "p0lose", "p0draw"):
    os.makedirs(os.path.join(GAME_GIF_DIR, sub), exist_ok=True)


# ================================================================
# Utility: load AZ models as in training
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
# Build AZ forward + MCTS selectors
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
    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
        del rng_key
        model_params, model_state = model
        current_player = state.current_player

        # board-only dynamics
        state = jax.vmap(_step_board)(state, action)

        # observation for side to move
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
    @jax.jit
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
# GateNet
# ================================================================

class GateNet(hk.Module):
    def __init__(self, num_options: int = 2, name: str = "GateNet"):
        super().__init__(name=name)
        self.num_options = num_options

    def __call__(self, obs, time_left_norm):
        """
        obs: (B,5,5,115)
        time_left_norm: (B,2)
        """
        x = obs.astype(jnp.float32)
        x = jnp.moveaxis(x, -1, 1)            # (B,115,5,5)

        conv = hk.Sequential([
            # FIX: padding must be 'VALID' or 'SAME'
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=64, kernel_shape=3, padding="SAME"),
            jax.nn.relu,
            hk.Flatten(),
        ])
        z = conv(x)

        z = jnp.concatenate([z, time_left_norm], axis=-1)

        h = hk.Linear(128)(z)
        h = jax.nn.relu(h)

        logits = hk.Linear(self.num_options)(h)
        value = hk.Linear(1)(h)[..., 0]
        return logits, value


def gate_forward_fn(obs_batch, time_batch):
    net = GateNet(num_options=2)
    return net(obs_batch, time_batch)


gate_forward = hk.without_apply_rng(hk.transform(gate_forward_fn))


# ================================================================
# Minimal Visualizer + GIF generator with clocks and side info
# ================================================================

@dataclass
class ColorSet:
    p1_color: str = "black"
    p2_color: str = "white"
    p1_outline: str = "black"
    p2_outline: str = "black"
    background_color: str = "white"
    grid_color: str = "black"
    text_color: str = "black"


class Visualizer:
    """
    Minimal visualizer for gardner_chess, with extra vertical space
    for side-to-move and clocks.
    """

    def __init__(self, scale: float = 1.0, color_theme: str = "light") -> None:
        self.config = {
            "GRID_SIZE": 50,
            "BOARD_WIDTH": 5,
            "BOARD_HEIGHT": 5,
            "SCALE": scale,
            "COLOR_SET": ColorSet(
                "none",
                "none",
                "gray",
                "white",
                "white" if color_theme == "light" else "#1e1e1e",
                "black" if color_theme == "light" else "silver",
                "black",
            ),
        }
        from pgx._src.dwg.gardner_chess import _make_gardner_chess_dwg
        self._make_dwg_group = _make_gardner_chess_dwg  # type: ignore

    def get_dwg(self, state: State):
        GRID_SIZE = self.config["GRID_SIZE"]
        BOARD_WIDTH = self.config["BOARD_WIDTH"]
        BOARD_HEIGHT = self.config["BOARD_HEIGHT"]
        SCALE = self.config["SCALE"]

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (BOARD_WIDTH + 1) * GRID_SIZE * SCALE,
                (BOARD_HEIGHT + 4) * GRID_SIZE * SCALE,  # extra vertical space
            ),
        )
        group = dwg.g()

        # background
        group.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE,
                    (BOARD_HEIGHT + 4) * GRID_SIZE,
                ),
                fill=self.config["COLOR_SET"].background_color,
            )
        )

        # board
        g = self._make_dwg_group(dwg, state, self.config)
        g.translate(
            GRID_SIZE * 0.5,
            GRID_SIZE * 0.5,
        )
        group.add(g)
        group.scale(SCALE)
        dwg.add(group)
        return dwg


def save_speed_gardner_gif_with_clocks(
    states: List[State],
    clocks: List[Tuple[float, float]],
    filename: str,
    white_player: int,
    color_theme: str = "light",
    scale: float = 1.0,
    frame_duration_seconds: float = 0.4,
) -> None:
    """
    states: list of State after each move
    clocks: list of (time_left_p0, time_left_p1)
    white_player: 0 or 1 (which RL player is white)
    """
    assert len(states) == len(clocks), "states and clocks must have same length"

    v = Visualizer(scale=scale, color_theme=color_theme)

    GRID_SIZE = v.config["GRID_SIZE"]
    BOARD_WIDTH = v.config["BOARD_WIDTH"]
    BOARD_HEIGHT = v.config["BOARD_HEIGHT"]
    text_color = v.config["COLOR_SET"].text_color

    frames = []

    white_str = f"P{white_player}"
    black_str = f"P{1 - white_player}"
    header_text = f"White: {white_str}  Black: {black_str}"

    for state, (t0, t1) in zip(states, clocks):
        dwg = v.get_dwg(state)
        group = dwg.elements[-1]  # board group

        # Board vertical extent ~ (0.5 .. 5.5)*GRID, so go well below.
        y_header = (BOARD_HEIGHT + 1.1) * GRID_SIZE
        y0 = (BOARD_HEIGHT + 1.9) * GRID_SIZE
        y1 = (BOARD_HEIGHT + 2.7) * GRID_SIZE

        # Smaller fonts so we fit everything
        header_font = GRID_SIZE * 0.55
        clock_font = GRID_SIZE * 0.5

        group.add(
            dwg.text(
                header_text,
                insert=(GRID_SIZE * 0.5, y_header),
                fill=text_color,
                font_size=header_font,
            )
        )
        group.add(
            dwg.text(
                f"P0 clock: {t0:.1f} ticks",
                insert=(GRID_SIZE * 0.5, y0),
                fill=text_color,
                font_size=clock_font,
            )
        )
        group.add(
            dwg.text(
                f"P1 clock: {t1:.1f} ticks",
                insert=(GRID_SIZE * 0.5, y1),
                fill=text_color,
                font_size=clock_font,
            )
        )

        # SVG → PNG in memory
        svg_bytes = dwg.tostring().encode("utf-8")
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
        img = imageio.imread(io.BytesIO(png_bytes))
        frames.append(img)

    imageio.mimsave(filename, frames, duration=frame_duration_seconds)


# ================================================================
# Jitted, single-game rollout + vmap over games
# ================================================================

def make_play_many_jitted(
    env_speed: GardnerChess,
    model_8,
    model_32,
    select_mcts_8,
    select_mcts_32,
    default_time: float,
    mode0: str,
    mode1: str,
):
    """
    mode0: 'gate' (player 0 = learned gate)
    mode1: 'gate' / 'always8' / 'always32' for player 1
    """
    default_time_f32 = jnp.float32(default_time)

    assert mode0 == "gate"
    assert mode1 in ("gate", "always8", "always32")

    if mode1 == "gate":
        mode1_code = 0
    elif mode1 == "always8":
        mode1_code = 1
    else:
        mode1_code = 2

    def play_one_game(gate_params, rng_key):
        state0 = env_speed.init(rng_key)

        def step_fn(carry, _):
            state, rng = carry
            rng, rng8, rng32 = jax.random.split(rng, 3)

            alive = ~(state.terminated | state.truncated)
            time_before = state.time_left
            cur = state.current_player

            my_time = jax.lax.select(cur == 0, time_before[0], time_before[1])
            opp_time = jax.lax.select(cur == 0, time_before[1], time_before[0])

            obs_b = state.observation[None, ...]
            time_norm_b = jnp.array(
                [[my_time / default_time_f32, opp_time / default_time_f32]],
                dtype=jnp.float32,
            )

            logits, _ = gate_forward.apply(gate_params, obs_b, time_norm_b)
            logits0 = logits[0]
            gate_choice = jnp.argmax(logits0)

            # P0 uses learned gate
            action_if0 = gate_choice

            # P1 uses whatever mode1 says
            if mode1_code == 0:
                action_if1 = gate_choice
            elif mode1_code == 1:
                action_if1 = jnp.int32(0)
            else:
                action_if1 = jnp.int32(1)

            gate_action = jax.lax.select(cur == 0, action_if0, action_if1)
            gate_action = jax.lax.select(alive, gate_action, jnp.int32(0))

            nsim = jax.lax.select(gate_action == 0, jnp.int32(8), jnp.int32(32))

            time_spent = jax.lax.select(
                alive,
                jax.lax.select(
                    gate_action == 0, jnp.int32(COST_8), jnp.int32(COST_32)
                ),
                jnp.int32(0),
            )

            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
            action_8 = select_mcts_8(model_8, state_b, rng8)[0]
            action_32 = select_mcts_32(model_32, state_b, rng32)[0]
            action = jax.lax.select(gate_action == 0, action_8, action_32)
            action = jax.lax.select(alive, action, jnp.int32(0))

            def do_step(s):
                return env_speed.step(s, (action, time_spent))

            state_next = jax.lax.cond(alive, do_step, lambda s: s, state)
            time_after = state_next.time_left
            done_after = state_next.terminated | state_next.truncated

            out = {
                "state": state_next,
                "player": cur,
                "gate_action": gate_action,
                "nsim": nsim,
                "time_before": time_before,
                "time_after": time_after,
                "move_mask": alive,
                "done": done_after,
            }
            return (state_next, rng), out

        (state_final, _), traj = jax.lax.scan(
            step_fn,
            (state0, rng_key),
            xs=None,
            length=MAX_TERMINATION_STEPS,
        )
        return traj, state_final

    @jax.jit
    def play_many(gate_params, rng_keys):
        trajs, finals = jax.vmap(play_one_game, in_axes=(None, 0))(gate_params, rng_keys)
        return trajs, finals

    return play_many


# ================================================================
# Main evaluation
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to play in parallel.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="always8",
        choices=["always8", "always32", "self"],
        help=(
            "Opponent gating policy for player 1:\n"
            "  always8  -> always pick nsim=8\n"
            "  always32 -> always pick nsim=32\n"
            "  self     -> also use learned gate"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    NUM_GAMES = args.num_games
    opponent = args.opponent
    seed = args.seed

    print(f"Running {NUM_GAMES} games, opponent='{opponent}', seed={seed}")

    # ----------------------
    # Load base AlphaZero models
    # ----------------------
    ckpt_paths = discover_checkpoints(CKPT_ROOT, BASE_ENV_ID, ITER_FILE)
    print("Found base checkpoints:", ckpt_paths)

    needed = [f"nsim_{n}" for n in BASE_NSIMS]
    for k in needed:
        if k not in ckpt_paths:
            raise RuntimeError(f"Missing checkpoint for {k} in {CKPT_ROOT}")

    env_id_8, cfg_8, model_8 = load_checkpoint(ckpt_paths["nsim_8"])
    env_id_32, cfg_32, model_32 = load_checkpoint(ckpt_paths["nsim_32"])
    print("Loaded base models:", env_id_8, env_id_32)

    if env_id_8 != BASE_ENV_ID or env_id_32 != BASE_ENV_ID:
        raise RuntimeError("Base model env_ids mismatch")

    if (
        cfg_8.num_layers != cfg_32.num_layers
        or cfg_8.num_channels != cfg_32.num_channels
        or cfg_8.resnet_v2 != cfg_32.resnet_v2
    ):
        raise RuntimeError("Base models have different architectures!")

    # ----------------------
    # Build speed env + forward + MCTS selectors
    # ----------------------
    env_speed = GardnerChess()
    rng = jax.random.PRNGKey(seed)

    forward = build_forward(env_speed, cfg_8)
    recurrent_fn_speed = make_recurrent_fn_speed(forward)
    select_mcts_8 = make_select_actions_mcts(forward, recurrent_fn_speed, 8)
    select_mcts_32 = make_select_actions_mcts(forward, recurrent_fn_speed, 32)

    # ----------------------
    # Load gate network params
    # ----------------------
    with open(GATE_CKPT_PATH, "rb") as f:
        gate_ckpt = pickle.load(f)
    if "gate_params" in gate_ckpt:
        gate_params = gate_ckpt["gate_params"]
    elif "params" in gate_ckpt:
        gate_params = gate_ckpt["params"]
    else:
        raise KeyError(
            f"Could not find 'gate_params' or 'params' in gate checkpoint {GATE_CKPT_PATH}. "
            f"Available keys: {list(gate_ckpt.keys())}"
        )
    print(
        f"Loaded gate network from {GATE_CKPT_PATH}, "
        f"update={gate_ckpt.get('update', 'unknown')}"
    )

    # default_time from env
    rng, key_init = jax.random.split(rng)
    tmp_state = env_speed.init(key_init)
    default_time = float(tmp_state.time_left[0])
    print("Default time (ticks):", default_time)

    # Opponent mode for player 1
    if opponent == "always8":
        mode1 = "always8"
    elif opponent == "always32":
        mode1 = "always32"
    else:
        mode1 = "gate"   # self-play

    play_many = make_play_many_jitted(
        env_speed=env_speed,
        model_8=model_8,
        model_32=model_32,
        select_mcts_8=select_mcts_8,
        select_mcts_32=select_mcts_32,
        default_time=default_time,
        mode0="gate",
        mode1=mode1,
    )

    # ----------------------
    # Run NUM_GAMES in parallel (vmapped)
    # ----------------------
    rng, key_games = jax.random.split(rng)
    rng_keys = jax.random.split(key_games, NUM_GAMES)  # (N,2)

    print("Compiling & running JAX simulation...")
    trajs, final_states = play_many(gate_params, rng_keys)
    trajs = jax.device_get(trajs)
    final_states = jax.device_get(final_states)

    # Unpack trajectory arrays
    players_all = np.array(trajs["player"])           # (N,T)
    nsim_all = np.array(trajs["nsim"])                # (N,T)
    move_mask_all = np.array(trajs["move_mask"])      # (N,T)
    time_before_all = np.array(trajs["time_before"])  # (N,T,2)
    time_after_all = np.array(trajs["time_after"])    # (N,T,2)
    done_all = np.array(trajs["done"])                # (N,T)

    states_traj = trajs["state"]

    # ----------------------
    # Aggregate stats
    # ----------------------
    total_wins_p0 = 0
    total_wins_p1 = 0
    total_draws = 0

    total_losses_p0 = 0
    total_losses_p0_timeout = 0
    total_losses_p0_checkmate = 0

    for g in range(NUM_GAMES):
        players_g = players_all[g]           # (T,)
        nsim_g = nsim_all[g]                 # (T,)
        move_mask_g = move_mask_all[g].astype(bool)
        time_before_g = time_before_all[g]   # (T,2)
        time_after_g = time_after_all[g]     # (T,2)
        done_g = done_all[g]                 # (T,)

        states_seq_g = jax.tree_util.tree_map(lambda x: x[g], states_traj)

        T = move_mask_g.shape[0]
        if done_g.any():
            last_step = int(np.argmax(done_g))
        else:
            last_step = T - 1

        valid_mask = move_mask_g & (np.arange(T) <= last_step)
        move_indices = np.where(valid_mask)[0]

        # final state + rewards
        final_state_g = jax.tree_util.tree_map(lambda x: x[g], final_states)
        rewards_vec = np.array(final_state_g.rewards, dtype=np.float32)
        r0 = float(rewards_vec[0])
        r1 = float(rewards_vec[1])

        # track losses by cause
        if r0 < 0.0:
            total_losses_p0 += 1
            time_left_p0 = float(final_state_g.time_left[0])
            if time_left_p0 <= 0.0:
                total_losses_p0_timeout += 1
            else:
                total_losses_p0_checkmate += 1

        if r0 > 0:
            total_wins_p0 += 1
            result_str = "P0 win"
            result_dir = os.path.join(GAME_GIF_DIR, "p0Win")
        elif r1 > 0:
            total_wins_p1 += 1
            result_str = "P1 win"
            result_dir = os.path.join(GAME_GIF_DIR, "p0lose")
        else:
            total_draws += 1
            result_str = "draw"
            result_dir = os.path.join(GAME_GIF_DIR, "p0draw")

        # Which RL player is white?
        # At t=0, side to move is white (turn=0), so players_g[0] is white.
        white_player = int(players_g[0])
        if white_player == 0:
            white_info = "White = P0, Black = P1"
        else:
            white_info = "White = P1, Black = P0"

        # Per-game nsim distribution (only over real moves)
        nsim_moves = nsim_g[move_indices]
        players_moves = players_g[move_indices]

        n8 = int((nsim_moves == 8).sum())
        n32 = int((nsim_moves == 32).sum())

        mask_p0 = players_moves == 0
        mask_p1 = players_moves == 1
        n8_p0 = int(((nsim_moves == 8) & mask_p0).sum())
        n32_p0 = int(((nsim_moves == 32) & mask_p0).sum())
        n8_p1 = int(((nsim_moves == 8) & mask_p1).sum())
        n32_p1 = int(((nsim_moves == 32) & mask_p1).sum())

        print(f"\n=== Game {g} ===")
        print(f"Result: {result_str}  (r0={r0}, r1={r1})")
        print(f"{white_info}")
        print(f"Total moves: {len(move_indices)}")
        print(f"nsim=8 moves:  {n8}  (P0: {n8_p0}, P1: {n8_p1})")
        print(f"nsim=32 moves: {n32}  (P0: {n32_p0}, P1: {n32_p1})")

        # Detailed per-move log
        print("Moves (idx, player, nsim, time_before -> time_after):")
        for local_idx, t in enumerate(move_indices):
            player_t = int(players_g[t])
            nsim_t = int(nsim_g[t])
            tb0, tb1 = time_before_g[t]
            ta0, ta1 = time_after_g[t]
            print(
                f"  {local_idx:02d}: P{player_t} nsim={nsim_t}"
                f"  [P0: {int(tb0):3d}->{int(ta0):3d}, P1: {int(tb1):3d}->{int(ta1):3d}]"
            )

        # Build frames for GIF (one frame per move, state AFTER the move)
        frame_states: List[State] = []
        frame_clocks: List[Tuple[float, float]] = []

        for t in move_indices:
            state_after_t = jax.tree_util.tree_map(lambda x: x[t], states_seq_g)
            frame_states.append(state_after_t)
            t0 = float(time_after_g[t, 0])
            t1 = float(time_after_g[t, 1])
            frame_clocks.append((t0, t1))

        gif_path = os.path.join(result_dir, f"game_{g:02d}.gif")
        if len(frame_states) > 0:
            save_speed_gardner_gif_with_clocks(
                frame_states,
                frame_clocks,
                gif_path,
                white_player=white_player,
                color_theme="light",
                scale=1.0,
                frame_duration_seconds=0.4,
            )
            print(f"Saved GIF animation for game {g} to {gif_path}")
        else:
            print(f"No moves in game {g}, skipping GIF.")

    # ----------------------
    # Overall summary
    # ----------------------
    print("\n================ OVERALL SUMMARY ================")
    print(f"Games played: {NUM_GAMES}")
    print(f"P0 (gate) wins        : {total_wins_p0}")
    print(f"P1 ({opponent}) wins  : {total_wins_p1}")
    print(f"Draws                 : {total_draws}")
    print(f"P0 losses total       : {total_losses_p0}")
    print(f"  - by timeout/flag   : {total_losses_p0_timeout}")
    print(f"  - by checkmate/else : {total_losses_p0_checkmate}")
    print("=================================================")


if __name__ == "__main__":
    main()
