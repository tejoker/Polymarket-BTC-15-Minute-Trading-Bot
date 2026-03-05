"""
Linear Upper Confidence Bound (LinUCB) Contextual Bandit
Evaluates the continuous 44-dimensional ChIMP execution vector as "context"
and predicts Epistemic Uncertainty. Overrides Maximum-Capital neural actions
with severe downside limits when entering unmapped market regimes.
"""
from typing import Tuple
import numpy as np
from loguru import logger

class LinUCBLayer:
    """
    Stochastic position-sizing cap. Learns a linear map between context X
    and abstract rewards while retaining an invertible Covariance matrix (A)
    to calculate the Upper Confidence Bound (exploration/uncertainty).
    """
    def __init__(self, context_dim: int = 44, alpha: float = 2.0):
        self.context_dim = context_dim
        # Exploration constraint (Higher alpha = wider bounds = faster cap triggers)
        self.alpha = alpha 
        
        # A matrix: Covariance (initialized to Identity)
        self.A = np.eye(self.context_dim)
        # b vector: Reward map (initialized to Zero)
        self.b = np.zeros(self.context_dim)
        
        # We cache the inverse periodically to prevent O(N^3) on the hotpath
        self._A_inv = np.eye(self.context_dim)
        self._updates_since_inv = 0
            
    def _update_inverse(self):
        """Update Sherman-Morrison cache for <5us evaluation."""
        self._A_inv = np.linalg.inv(self.A)
        self._updates_since_inv = 0

    def observe_and_cap(self, context_vector: np.ndarray, requested_size: float) -> Tuple[float, float, bool]:
        """
        Takes the current market state and the IPPO Agent's requested position
        size (0.0 to 1.0). If the Bandit detects high Epistemic Uncertainty,
        it instantly overrides the magnitude downward.
        
        Returns:
            (bounded_size, uncertainty_score, was_capped_bool)
        """
        if context_vector.shape[0] != self.context_dim:
            # Fallback to extreme safety if dimensions mismatch
            return 0.05, 1.0, True
            
        # Ensure context vector is a column
        x = context_vector.reshape(-1)
        
        # Calculate Expected Payoff (theta^T * x)
        theta = self._A_inv @ self.b
        expected_payoff = theta.T @ x
        
        # Calculate Epistemic Uncertainty (x^T * A^-1 * x)
        uncertainty = np.sqrt(x.T @ self._A_inv @ x)
        
        # Upper Confidence Bound
        ucb_score = expected_payoff + (self.alpha * uncertainty)
        
        # The core cybernetic logic:
        # If uncertainty is massive (we are in an unmapped regime, possibly flash crash)
        # We CAP the maximum allowable size inversely to the uncertainty.
        # e.g., if uncertainty = 2.0 -> max_cap = 1 / 2.0 = 0.50 (50% max position)
        cap = 1.0
        if uncertainty > 0.5:
            cap = min(1.0, (1.0 / (uncertainty * 2.0)))
            
        was_capped = False
        final_size = requested_size
        
        if requested_size > cap:
            logger.warning(
                f"🛡️  LinUCB Bandit override! 📈 Uncertainty: {uncertainty:.2f} | "
                f"Agent Requested: {requested_size*100:.1f}% -> Capped at: {cap*100:.1f}%"
            )
            final_size = cap
            was_capped = True
            
        return final_size, uncertainty, was_capped

    def step_learn(self, context_vector: np.ndarray, continuous_reward: float):
        """
        Online Ridge-Regression update. Call this after the trade resolves
        or alongside the JAX Environment sweeps to teach the Bandit.
        """
        x = context_vector.reshape(-1)
        
        # Update Covariance (A = A + x*x^T)
        self.A += np.outer(x, x)
        
        # Update reward mapping (b = b + reward*x)
        self.b += continuous_reward * x
        
        self._updates_since_inv += 1
        if self._updates_since_inv > 50:
            self._update_inverse()
