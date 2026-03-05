"""
FTRL-Proximal (Follow-The-Regularized-Leader)
Replaces exponential moving average (EMA) learning with online convex optimization.
Provides mathematically proven regret bounds in adversarial, non-stationary market regimes.
"""
import math

class FTRLProximal:
    """
    Online Learning Algorithm for component weight optimization.
    """
    def __init__(self, alpha: float = 0.1, beta: float = 1.0, l1: float = 0.1, l2: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        
        # State tracking per feature (signal source)
        self.z = {}
        self.n = {}
        
    def get_weight(self, feature_id: str) -> float:
        """Calculate current weight for a feature."""
        z_i = self.z.get(feature_id, 0.0)
        n_i = self.n.get(feature_id, 0.0)
        
        # Determine weight with L1/L2 regularization
        sign_z = -1.0 if z_i < 0 else 1.0
        
        # L1 thresholding (encourages sparsity if unhelpful)
        if abs(z_i) <= self.l1:
            return 0.0
            
        w = (sign_z * self.l1 - z_i) / ((self.beta + math.sqrt(n_i)) / self.alpha + self.l2)
        return w

    def update(self, feature_id: str, loss_gradient: float):
        """
        Update state using the observed loss gradient.
        In our context, negative gradient means the signal performed well
        (decreasing loss), and positive gradient means it performed poorly.
        """
        z_i = self.z.get(feature_id, 0.0)
        n_i = self.n.get(feature_id, 0.0)
        
        current_w = self.get_weight(feature_id)
        
        # Avoid exploding gradients
        gradient_clipped = max(-10.0, min(10.0, loss_gradient))
        
        sigma = (math.sqrt(n_i + gradient_clipped**2) - math.sqrt(n_i)) / self.alpha
        
        self.z[feature_id] = z_i + gradient_clipped - sigma * current_w
        self.n[feature_id] = n_i + gradient_clipped**2

    def get_normalized_weights(self, feature_ids: list[str]) -> dict[str, float]:
        """
        Calculates Softmax over the FTRL weights to generate valid weight distributions
        for the Signal Fusion module.
        """
        raw_weights = {fid: self.get_weight(fid) for fid in feature_ids}
        
        # If all weights are zero (e.g. at start), return uniform
        if all(w == 0 for w in raw_weights.values()) and raw_weights:
            uniform = 1.0 / len(raw_weights)
            return {fid: uniform for fid in feature_ids}
            
        # Softmax to normalize between 0 and 1
        max_w = max(raw_weights.values())
        exp_weights = {fid: math.exp(w - max_w) for fid, w in raw_weights.items()}
        sum_exp = sum(exp_weights.values())
        
        return {fid: ew / sum_exp for fid, ew in exp_weights.items()}
