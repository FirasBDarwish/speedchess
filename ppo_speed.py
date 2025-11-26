import sys
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import pgx
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
class Config:
    """Hyperparameters for Speed Chess PPO."""
    LR = 2.5e-4
    NUM_ENVS = 256            # Parallel games (Batch size)
    NUM_STEPS = 128           # Steps per rollout
    TOTAL_TIMESTEPS = 5e6     # Total frames to train on
    UPDATE_EPOCHS = 4         # PPO update epochs
    NUM_MINIBATCHES = 4       # Minibatches per epoch
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    # Speed Chess Specifics
    # We force the agent to "spend" this much time per move.
    # If the game is 1000 ticks, 10 ticks/move means ~100 moves max budget.
    TIME_PER_MOVE = 10        

# -------------------------------------------------------------------
# 2. ENVIRONMENT WRAPPER
# -------------------------------------------------------------------
class SpeedChessWrapper:
    """
    Wraps the PGX Chess environment to handle Speed Chess time controls.
    
    The raw environment requires step(state, (action, time_spent)).
    PPO only outputs 'action'. This wrapper injects a fixed 'time_spent'
    so the agent learns to play under time pressure constraints.
    """
    def __init__(self):
        # Import your local modified chess env or generic pgx chess
        # Assuming pgx.make("chess") loads your modified class or we use the class directly:
        from pgx.chess import Chess
        self._env = Chess()

    @property
    def num_actions(self):
        return 4672  # Standard AlphaZero/PGX chess action space

    @property
    def observation_shape(self):
        return (8, 8, 119)

    def init(self, key):
        return self._env.init(key)

    def step(self, state, action, key):
        # INJECT TIME SPENT HERE
        # We construct the tuple (action, time_spent) expected by your modified env
        time_spent = jnp.int32(Config.TIME_PER_MOVE)
        action_input = (action, time_spent)
        
        return self._env.step(state, action_input, key)

    def observe(self, state):
        return self._env.observe(state)

# -------------------------------------------------------------------
# 3. NETWORK (Actor-Critic)
# -------------------------------------------------------------------
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (..., 8, 8, 119)
        
        # Simple ResNet-like or Conv block for Chess
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(512)(x)
        x = nn.relu(x)

        # Actor Head
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        
        # Critic Head
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        return actor_mean, jnp.squeeze(critic, axis=-1)

# -------------------------------------------------------------------
# 4. TRAINING STATE & TRANSITIONS
# -------------------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: Any

# -------------------------------------------------------------------
# 5. TRAINING LOOP
# -------------------------------------------------------------------
def make_train(config):
    env = SpeedChessWrapper()
    
    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.num_actions)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_shape))
        network_params = network.init(_rng, init_x)

        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(config.LR, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        # vmap the env.init -> (NUM_ENVS, State)
        env_state = jax.vmap(env.init)(reset_rng)

        # -----------------------------------------------------------
        # TRAIN STEP (Rollout + Update)
        # -----------------------------------------------------------
        def _update_step(runner_state, _):
            train_state, env_state, rng = runner_state

            # --- ROLLOUT PHASE ---
            def _env_step(runner_state, _):
                env_state, rng = runner_state
                
                # Select Action
                rng, _rng = jax.random.split(rng)
                
                # FIX 1: vmap the observe call because env_state is batched (NUM_ENVS)
                obs = jax.vmap(env.observe)(env_state)
                
                logits, value = network.apply(train_state.params, obs)
                
                # Mask illegal moves (very important for chess)
                legal_mask = env_state.legal_action_mask
                logits = logits + jnp.where(legal_mask, 0.0, -1e9)
                
                # Manual Sampling
                action = jax.random.categorical(_rng, logits)
                # Compute log_prob manually using log_softmax
                log_probs = jax.nn.log_softmax(logits)
                log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)

                # Step Env (vmap is already handled here by explicit call)
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, config.NUM_ENVS)
                
                # We use vmap to step all environments in parallel
                next_env_state = jax.vmap(env.step)(env_state, action, step_rngs)

                # FIX 3: Correct Reward Indexing
                # Select the reward for the player who acted (env_state.current_player)
                # Using advanced indexing: rewards[arange(N), current_player]
                # This avoids the (256, 256) cross-product shape mismatch
                rewards = next_env_state.rewards[jnp.arange(config.NUM_ENVS), env_state.current_player]

                transition = Transition(
                    done=next_env_state.terminated,
                    action=action,
                    value=value,
                    reward=rewards, 
                    log_prob=log_prob,
                    observation=obs,
                    info={},
                )
                return (next_env_state, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (env_state, rng), None, length=config.NUM_STEPS
            )
            env_state, rng = runner_state

            # --- GAE CALCULATION ---
            # FIX 2: vmap the observe call here as well
            last_obs = jax.vmap(env.observe)(env_state)
            
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    
                    delta = reward + config.GAMMA * next_value * (1 - done) - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # --- UPDATE PHASE ---
            def _loss_fn(params, traj_batch, gae, targets):
                # Rerun network to get current logits/values
                logits, value = network.apply(params, traj_batch.observation)
                
                # Manual LogProb / Entropy
                log_probs = jax.nn.log_softmax(logits)
                log_prob = jnp.take_along_axis(log_probs, traj_batch.action[..., None], axis=-1).squeeze(-1)
                
                # Entropy approx
                probs = jnp.exp(log_probs)
                entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                
                loss_actor1 = ratio * gae
                loss_actor2 = jnp.clip(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * gae
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                
                loss_critic = jnp.square(value - targets).mean()
                
                total_loss = loss_actor + config.VF_COEF * loss_critic - config.ENT_COEF * entropy
                return total_loss, (loss_actor, loss_critic, entropy)

            # Flatten batch for SGD
            batch_size = config.NUM_ENVS * config.NUM_STEPS
            minibatch_size = batch_size // config.NUM_MINIBATCHES
            
            # Reshape data
            def flatten(x):
                return x.reshape((batch_size,) + x.shape[2:])
            
            traj_batch = jax.tree_util.tree_map(flatten, traj_batch)
            advantages = flatten(advantages)
            targets = flatten(targets)

            # FIX 4: Pass rng through the epoch scan carry to avoid UnboundLocalError
            def _update_epoch(carry, _):
                train_state, rng = carry
                
                # Use split for new permutation key
                rng, perm_rng = jax.random.split(rng)
                permutation = jax.random.permutation(perm_rng, batch_size)
                
                # FIX 5: Reshape logic to avoid dynamic slicing error
                # Instead of slicing by index inside the scan, we permute and reshape
                # the data beforehand so we can scan over it directly.
                
                def _reshape_for_scan(x):
                    # Shuffle
                    shuffled = x[permutation]
                    # Reshape to (NUM_MINIBATCHES, MINIBATCH_SIZE, ...)
                    return shuffled.reshape((config.NUM_MINIBATCHES, minibatch_size) + x.shape[1:])

                # Prepare minibatches
                batch_traj = jax.tree_util.tree_map(_reshape_for_scan, traj_batch)
                batch_adv = _reshape_for_scan(advantages)
                batch_tar = _reshape_for_scan(targets)
                
                def _update_minibatch(train_state, minibatch_data):
                    # Unpack the scanned data item
                    batch, batch_gae, batch_targets = minibatch_data
                    
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, metrics), grads = grad_fn(train_state.params, batch, batch_gae, batch_targets)
                    
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, metrics

                # Scan over the prepared minibatch data
                train_state, metrics = jax.lax.scan(
                    _update_minibatch, 
                    train_state, 
                    (batch_traj, batch_adv, batch_tar)
                )
                return (train_state, rng), metrics

            # Run epochs with threaded rng
            (train_state, rng), metrics = jax.lax.scan(
                _update_epoch, (train_state, rng), None, length=config.UPDATE_EPOCHS
            )
            
            metric_tree = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
            # Return the updated rng from the epochs to the main loop state
            return (train_state, env_state, rng), metric_tree

        # -----------------------------------------------------------
        # MAIN LOOP SCAN
        # -----------------------------------------------------------
        num_updates = int(config.TOTAL_TIMESTEPS // config.NUM_ENVS // config.NUM_STEPS)
        runner_state = (train_state, env_state, rng)
        
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=num_updates
        )
        
        return runner_state, metrics

    return train

# -------------------------------------------------------------------
# 6. RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"JAX Device: {jax.devices()}")
    
    config = Config()
    rng = jax.random.PRNGKey(42)
    
    train_jit = jax.jit(make_train(config))
    
    print("Starting training...")
    # This will compile first (may take a minute) then run very fast
    final_state, metrics = train_jit(rng)
    
    print("Training Complete.")
    print(f"Final Actor Loss: {metrics[0][-1]:.4f}")
    print(f"Final Critic Loss: {metrics[1][-1]:.4f}")
    print(f"Final Entropy: {metrics[2][-1]:.4f}")