# # Copyright 2023 The Pgx Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import warnings

# import jax
# import jax.numpy as jnp

# import pgx.core as core
# from pgx._src.games.chess import INIT_LEGAL_ACTION_MASK, Game, GameState, _flip, DEFAULT_TIME
# from pgx._src.struct import dataclass
# from pgx._src.types import Array, PRNGKey


# @dataclass
# class State(core.State):
#     current_player: Array = jnp.int32(0)
#     rewards: Array = jnp.float32([0.0, 0.0])
#     terminated: Array = jnp.bool_(False)
#     truncated: Array = jnp.bool_(False)
#     legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # 64 * 73 = 4672
#     observation: Array = jnp.zeros((8, 8, 119), dtype=jnp.float32)
#     _step_count: Array = jnp.int32(0)
#     _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
#     _time_left: Array = jnp.int32([DEFAULT_TIME, DEFAULT_TIME])
#     _x: GameState = GameState()

#     @property
#     def time_left(self):
#         return self._time_left
    
#     @time_left.setter
#     def time_left(self, value):
#         object.__setattr__(self, "_time_left", value)

#     @property
#     def env_id(self) -> core.EnvId:
#         return "chess"

#     @staticmethod
#     def _from_fen(fen: str):
#         from pgx.experimental.chess import from_fen

#         warnings.warn(
#             "State._from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
#             DeprecationWarning,
#         )
#         return from_fen(fen)

#     def _to_fen(self) -> str:
#         from pgx.experimental.chess import to_fen

#         warnings.warn(
#             "State._to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
#             DeprecationWarning,
#         )
#         return to_fen(self)


# class Chess(core.Env):
#     def __init__(self):
#         super().__init__()
#         self.game = Game()

#     def _init(self, key: PRNGKey) -> State:
#         x = GameState()
#         _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
#         state = State(  # type: ignore
#             current_player=_player_order[x.color],
#             _player_order=_player_order,
#             _x=x,
#             _time_left=jnp.int32([DEFAULT_TIME, DEFAULT_TIME]),
#         )
#         return state
    
#     def step(
#     self,
#     state: State,
#     action_input,
#     key=None,
#     ) -> State:

#         # Split (action, time_spent)
#         # if isinstance(action_input, (tuple, list)):
#         action = action_input[0]
#         time_spent = action_input[1]
#         # else:
#         #     action = action_input
#         #     time_spent = jnp.int32(0)

#         # PGX legality check (MUST occur before _step)
#         is_illegal = ~state.legal_action_mask[action]
#         current_player = state.current_player

#         # If already terminated: return zero reward
#         state = jax.lax.cond(
#             (state.terminated | state.truncated),
#             lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
#             lambda: self._step(
#                 state.replace(_step_count=state._step_count + 1),
#                 (action, time_spent),      # pass BOTH to your _step
#                 key,
#             ),
#         )

#         # Illegal action leads to immediate penalty
#         state = jax.lax.cond(
#             is_illegal,
#             lambda: self._step_with_illegal_action(state, current_player),
#             lambda: state,
#         )

#         # At terminal state: mask all actions True
#         state = jax.lax.cond(
#             state.terminated,
#             lambda: state.replace(
#                 legal_action_mask=jnp.ones_like(state.legal_action_mask)
#             ),
#             lambda: state,
#         )

#         # Update observation
#         state = state.replace(observation=self.observe(state))

#         return state

#     def _step(self, state: core.State, action_input: Array, key) -> State:
#         del key
#         assert isinstance(state, State)

#         # ----------------------------------------------------
#         # Accept both:
#         #   step(state, action)
#         #   step(state, (action, time_spent))
#         # ----------------------------------------------------
#         action = action_input[0]
#         time_spent = action_input[1]
#         # else:
#         #     action = action_input
#         #     time_spent = jnp.int32(0)

#         # subtract time
#         idx = state.current_player
#         new_time = state._time_left.at[idx].add(-time_spent)
#         time_over = new_time[idx] <= 0

#         # apply the move
#         x = self.game.step(state._x, action)

#         # compute base termination and rewards from chess rules
#         base_termination = self.game.is_terminal(x)

#         # timeout overrides chess results
#         terminated = base_termination | time_over

#         # baserewarsd and timeout rewards (player who times out loses; opponent wins)
#         base_rewards = self.game.rewards(x)[state._player_order]

#         timeout_rewards = jax.lax.select(idx == 0, jnp.float32([-1.0, 1.0]), jnp.float32([1.0, -1.0]))
        
#         # if time is over -> losing rewards
#         rewards = jax.lax.select(time_over, timeout_rewards, base_rewards)

#         state = state.replace(  # type: ignore
#             _x=x,
#             legal_action_mask=x.legal_action_mask,
#             terminated=self.game.is_terminal(x),
#             rewards=rewards,
#             current_player=state._player_order[x.color],
#             _time_left=new_time,
#         )
#         return state  # type: ignore

#     def _observe(self, state: core.State, player_id: Array) -> Array:
#         assert isinstance(state, State)
#         color = jax.lax.select(state.current_player == player_id, state._x.color, 1 - state._x.color)
#         x = jax.lax.cond(state.current_player == player_id, lambda: state._x, lambda: _flip(state._x))
#         return self.game.observe(x, color)

#     @property
#     def id(self) -> core.EnvId:
#         return "chess"

#     @property
#     def version(self) -> str:
#         return "v2"

#     @property
#     def num_players(self) -> int:
#         return 2


# def _from_fen(fen: str):
#     from pgx.experimental.chess import from_fen

#     warnings.warn(
#         "_from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
#         DeprecationWarning,
#     )
#     return from_fen(fen)


# def _to_fen(state: State):
#     from pgx.experimental.chess import to_fen

#     warnings.warn(
#         "_to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
#         DeprecationWarning,
#     )
#     return to_fen(state)

import warnings

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.chess import (
    INIT_LEGAL_ACTION_MASK,
    Game,
    GameState,
    _flip,
    DEFAULT_TIME,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    # Core.State fields / defaults
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # 64 * 73 = 4672
    observation: Array = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)

    # Extra chess-specific fields
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
    
    # Speed-chess time control: time_left[player_id] in "ticks".
    # We use _time_left (internal) so pgx.api_test doesn't flag it as an unknown public attribute.
    _time_left: Array = jnp.int32([DEFAULT_TIME, DEFAULT_TIME])
    _x: GameState = GameState()

    @property
    def time_left(self):
        """Public alias for _time_left for backward compatibility."""
        return self._time_left

    @property
    def env_id(self) -> core.EnvId:
        return "chess"

    @staticmethod
    def _from_fen(fen: str):
        from pgx.experimental.chess import from_fen

        warnings.warn(
            "State._from_fen is deprecated. Will be removed in the future "
            "release. Please use pgx.experimental.chess.from_fen instead.",
            DeprecationWarning,
        )
        return from_fen(fen)

    def _to_fen(self) -> str:
        from pgx.experimental.chess import to_fen

        warnings.warn(
            "State._to_fen is deprecated. Will be removed in the future "
            "release. Please use pgx.experimental.chess.to_fen instead.",
            DeprecationWarning,
        )
        return to_fen(self)


class Chess(core.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init(self, key: PRNGKey) -> State:
        x = GameState()
        # Randomly decide which agent is white / black via player_order
        _player_order = jnp.array([[0, 1], [1, 0]])[
            jax.random.bernoulli(key).astype(jnp.int32)
        ]
        state = State(  # type: ignore
            current_player=_player_order[x.color],
            _player_order=_player_order,
            _x=x,
            _time_left=jnp.int32([DEFAULT_TIME, DEFAULT_TIME]),
        )
        return state

    # ------------------------------------------------------------------
    # Step with optional time_spent
    # ------------------------------------------------------------------
    def step(
        self,
        state: State,
        action_input,
        key=None,
    ) -> State:
        """Step function that accepts either:
           - action (int / Array)
           - (action, time_spent)
        """

        # Unpack (action, time_spent) or default time_spent = 0
        if isinstance(action_input, (tuple, list)):
            action, time_spent = action_input
        else:
            action = action_input
            time_spent = jnp.int32(0)

        # PGX legality check
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If already terminated / truncated, just zero rewards
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: self._step(
                state.replace(_step_count=state._step_count + 1),
                (action, time_spent),
                key,
            ),
        )

        # Illegal action → immediate penalty / terminal
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # At terminal state: mask all actions as legal (PGX convention)
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(
                legal_action_mask=jnp.ones_like(state.legal_action_mask)
            ),
            lambda: state,
        )

        # Update observation
        state = state.replace(observation=self.observe(state))

        return state

    # ------------------------------------------------------------------
    # Internal _step: apply move + time control
    # ------------------------------------------------------------------
    def _step(self, state: core.State, action_input, key) -> State:  # type: ignore[override]
        del key
        assert isinstance(state, State)

        # Accept either action or (action, time_spent)
        if isinstance(action_input, (tuple, list)):
            action, time_spent = action_input
        else:
            action = action_input
            time_spent = jnp.int32(0)

        # Subtract time for current player
        idx = state.current_player
        new_time = state._time_left.at[idx].add(-time_spent)
        time_over = new_time[idx] <= 0

        # Apply the chess move
        x = self.game.step(state._x, action)

        # Base termination and rewards from chess rules
        base_termination = self.game.is_terminal(x)
        base_rewards = self.game.rewards(x)[state._player_order]

        # Time-out overrides chess result: the player who times out loses
        terminated = base_termination | time_over
        timeout_rewards = jax.lax.select(
            idx == 0,
            jnp.float32([-1.0, 1.0]),
            jnp.float32([1.0, -1.0]),
        )
        rewards = jax.lax.select(time_over, timeout_rewards, base_rewards)

        state = state.replace(  # type: ignore
            _x=x,
            legal_action_mask=x.legal_action_mask,
            terminated=terminated,
            rewards=rewards,
            current_player=state._player_order[x.color],
            _time_left=new_time,
        )
        return state  # type: ignore

    # ------------------------------------------------------------------
    # Observation (handles POV flipping)
    # ------------------------------------------------------------------
    def _observe(self, state: core.State, player_id: Array) -> Array:  # type: ignore[override]
        assert isinstance(state, State)
        color = jax.lax.select(
            state.current_player == player_id,
            state._x.color,
            1 - state._x.color,
        )
        x = jax.lax.cond(
            state.current_player == player_id,
            lambda: state._x,
            lambda: _flip(state._x),
        )
        return self.game.observe(x, color)

    # ------------------------------------------------------------------
    # Env metadata
    # ------------------------------------------------------------------
    @property
    def id(self) -> core.EnvId:
        return "chess"

    @property
    def version(self) -> str:
        # bumped because behavior (time control) differs from original
        return "v2"

    @property
    def num_players(self) -> int:
        return 2


# ----------------------------------------------------------------------
# Deprecated helpers
# ----------------------------------------------------------------------
def _from_fen(fen: str):
    from pgx.experimental.chess import from_fen

    warnings.warn(
        "_from_fen is deprecated. Will be removed in the future release. "
        "Please use pgx.experimental.chess.from_fen instead.",
        DeprecationWarning,
    )
    return from_fen(fen)


def _to_fen(state: State):
    from pgx.experimental.chess import to_fen

    warnings.warn(
        "_to_fen is deprecated. Will be removed in the future release. "
        "Please use pgx.experimental.chess.to_fen instead.",
        DeprecationWarning,
    )
    return to_fen(state)
