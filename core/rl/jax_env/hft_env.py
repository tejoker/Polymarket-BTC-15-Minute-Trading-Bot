"""
JaxMARL-HFT Environment
A GPU-accelerated Limit Order Book (LOB) Multi-Agent Reinforcement Learning
environment built entirely in JAX. Compiles to XLA for sub-microsecond vectorization.
"""
from typing import Dict, Tuple, Any

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit
except ImportError:
    jax = None
    jnp = None
    vmap = None
    jit = None

from loguru import logger

class JaxMARLHFTEnv:
    """
    Simulates a stochastic Limit Order Book environment for HFT RL training.
    Agents interact by placing, canceling, or executing orders.
    """
    def __init__(
        self,
        num_levels: int = 10,
        max_steps: int = 500,
        tick_size: float = 0.001
    ):
        self.num_levels = num_levels
        self.max_steps = max_steps
        self.tick_size = tick_size
        
        if jax is None:
            logger.warning("JAX is not installed. GPU-accelerated MARL cannot run.")
            
    # --- Environment Dynamics (Jittable) ---
    
    @staticmethod
    def _get_initial_state(rng_key: Any, num_levels: int) -> Dict[str, Any]:
        """Generate a synthetic starting limit order book state."""
        if jnp is None: return {}
        
        k1, k2, k3 = jax.random.split(rng_key, 3)
        mid_price = 0.50 # Polymarket probability domain
        
        # Bids: [0.49, 0.48, ...]
        bids_prices = mid_price - jnp.arange(1, num_levels + 1) * 0.001
        bids_volumes = jax.random.uniform(k1, shape=(num_levels,), minval=100.0, maxval=5000.0)
        
        # Asks: [0.51, 0.52, ...]
        asks_prices = mid_price + jnp.arange(1, num_levels + 1) * 0.001
        asks_volumes = jax.random.uniform(k2, shape=(num_levels,), minval=100.0, maxval=5000.0)
        
        return {
            "step": jnp.int32(0),
            "bids_prices": bids_prices,
            "bids_volumes": bids_volumes,
            "asks_prices": asks_prices,
            "asks_volumes": asks_volumes,
            "inventory": jnp.zeros((2,), dtype=jnp.float32), # [Agent0 (Alpha), Agent1 (Execution)]
            "cash": jnp.zeros((2,), dtype=jnp.float32),
            "mid_price_history": jnp.ones((20,)) * mid_price,
            # DSR Trackers
            "dsr_ema_return": jnp.float32(0.0),
            "dsr_ema_variance": jnp.float32(0.0001)
        }

    # Use JAX decorator for XLA compilation if available
    def reset(self, rng_key: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment state."""
        if jnp is None: return {}, {}
        state = self._get_initial_state(rng_key, self.num_levels)
        obs = self._get_obs(state)
        return obs, state

    def _get_obs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract observation vector for the IPPO agents."""
        if jnp is None: return {}
        # Observation is the flattened orderbook depth + agent inventory / cash
        obs_alpha = jnp.concatenate([
            state["bids_prices"], state["bids_volumes"],
            state["asks_prices"], state["asks_volumes"],
            jnp.array([state["inventory"][0], state["cash"][0]]),
            state["mid_price_history"]
        ])
        
        obs_exec = jnp.concatenate([
            state["bids_prices"], state["bids_volumes"],
            state["asks_prices"], state["asks_volumes"],
            jnp.array([state["inventory"][1], state["cash"][1]]),
            state["mid_price_history"]
        ])
        
        return {
            "alpha_agent": obs_alpha,
            "execution_agent": obs_exec
        }

    def step(self, state: Dict[str, Any], actions: Dict[str, Any], rng_key: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step of the environment (vectorizable).
        actions["alpha_agent"]: continuous action [-1, 1] for target position.
        actions["execution_agent"]: continuous action [0, 1] for limit order depth fraction.
        """
        if jnp is None: return {}, {}, {}, {}, {}
        
        # Advance state step
        new_step = state["step"] + 1
        
        # Mock Market Dynamics (Stochastic order flow crossing the spread)
        k_flow, k_vol = jax.random.split(rng_key, 2)
        market_flow = jax.random.normal(k_flow) * 0.005 # Random price drift
        
        # Shift Mid Price
        current_mid = state["asks_prices"][0] - (state["asks_prices"][0] - state["bids_prices"][0])/2.0
        new_mid = current_mid + market_flow
        
        # Update history
        new_history = jnp.roll(state["mid_price_history"], shift=1)
        new_history = new_history.at[0].set(new_mid)
        
        # Process Actions
        alpha_action = actions.get("alpha_agent", jnp.array(0.0))
        exec_action = actions.get("execution_agent", jnp.array(0.0))
        
        # Alpha agent attempts to acquire position target
        target_inventory = alpha_action * 1000.0 # Max $1000 pos
        inventory_delta = target_inventory - state["inventory"][0]
        
        # Execution agent decides *how* to acquire it (simulated slippage penalty)
        # Exec action close to 0: aggressive market order (high slippage)
        # Exec action close to 1: passive limit maker (low slippage, execution risk)
        
        slippage_penalty = (1.0 - jnp.abs(exec_action)) * jnp.abs(inventory_delta) * 0.005
        
        # === Compute Rewards (Phase 9: Differential Sharpe Ratio) ===
        # Exponential moving decay for online variance tracking (decay rate = 0.05)
        decay = 0.05
        
        # Absolute trade yield
        alpha_pnl = state["inventory"][0] * market_flow 
        alpha_raw_reward = alpha_pnl - (jnp.abs(inventory_delta) * 0.001) # Transaction fee
        
        # Update EMA Return and Variance
        new_dsr_return = (1.0 - decay) * state["dsr_ema_return"] + decay * alpha_raw_reward
        pnl_diff = alpha_raw_reward - new_dsr_return
        new_dsr_variance = (1.0 - decay) * state["dsr_ema_variance"] + decay * (pnl_diff ** 2)
        
        # Differential Sharpe Ratio calculation
        # DSR = (Return(t) - EMA_Return) / StdDev - (Return(t)**2 - EMA_Variance) / (2 * StdDev**3)
        std_dev = jnp.sqrt(new_dsr_variance) + 1e-8
        dsr_term_1 = (alpha_raw_reward - new_dsr_return) / std_dev
        dsr_term_2 = ((alpha_raw_reward**2) - new_dsr_variance) / (2.0 * (std_dev**3))
        
        alpha_reward = dsr_term_1 - dsr_term_2
        
        # Exec agent is penalized purely on slippage 
        exec_reward = -slippage_penalty
        
        # Update States
        new_inventory = state["inventory"].at[0].set(target_inventory)
        new_cash = state["cash"].at[0].set(state["cash"][0] - (inventory_delta * new_mid) - slippage_penalty)
        
        next_state = {
            "step": new_step,
            "bids_prices": jnp.clip(state["bids_prices"] + market_flow, 0.01, 0.99),
            "bids_volumes": state["bids_volumes"], # Unchanged simplified
            "asks_prices": jnp.clip(state["asks_prices"] + market_flow, 0.01, 0.99),
            "asks_volumes": state["asks_volumes"],
            "inventory": new_inventory,
            "cash": new_cash,
            "mid_price_history": new_history,
            "dsr_ema_return": new_dsr_return,
            "dsr_ema_variance": new_dsr_variance
        }
        
        rewards = {"alpha_agent": alpha_reward, "execution_agent": exec_reward}
        dones = {"__all__": new_step >= self.max_steps}
        infos = {}
        
        return self._get_obs(next_state), next_state, rewards, dones, infos
