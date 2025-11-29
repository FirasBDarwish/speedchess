import jax
import jax.numpy as jnp
from speed_gardner_chess import GardnerChess
import pgx


def _first_legal_action(mask: jnp.ndarray) -> jnp.ndarray:
    # mask is shape (25 * 49,)
    return jnp.argmax(mask.astype(jnp.int32))


def test_time_spent_decreases_clock_without_timeout():
    env = GardnerChess()
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    cur = int(state.current_player)
    start_time = int(state.time_left[cur])

    action = _first_legal_action(state.legal_action_mask)
    time_spent = jnp.int32(5)

    next_state = env.step(state, (action, time_spent))

    # Clock should decrease only for the player who moved
    assert int(next_state.time_left[cur]) == start_time - int(time_spent)
    assert int(next_state.time_left[1 - cur]) == int(state.time_left[1 - cur])
    print("test_time_spent_decreases_clock_without_timeout time_left passed")
    # And the game should normally still be running
    assert not bool(next_state.terminated)
    print("test_time_spent_decreases_clock_without_timeout terminated passed")


def test_timeout_leads_to_loss_for_moving_player():
    env = GardnerChess()
    key = jax.random.PRNGKey(1)
    state = env.init(key)

    cur = int(state.current_player)
    action = _first_legal_action(state.legal_action_mask)

    # Spend more time than we have on the clock
    overspend = int(state.time_left[cur]) + 1
    next_state = env.step(state, (action, jnp.int32(overspend)))

    assert bool(next_state.terminated)
    print("test_timeout_leads_to_loss_for_moving_player terminated passed")
    rewards = next_state.rewards
    # Player who flagged loses
    assert float(rewards[cur]) == -1.0
    assert float(rewards[1 - cur]) == 1.0
    print("test_timeout_leads_to_loss_for_moving_player rewards passed")


if __name__ == "__main__":
    test_time_spent_decreases_clock_without_timeout()
    test_timeout_leads_to_loss_for_moving_player()
    print("All tests passed")