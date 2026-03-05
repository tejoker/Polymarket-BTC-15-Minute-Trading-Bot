"""
Kelly Criterion Extension (KCE) & Avellaneda-Stoikov
Replaces static arbitrary risk matrices with mathematically optimal 
dynamic position sizing based on execution confidence, inventory, and volatility.
"""

import math
import logging

logger = logging.getLogger(__name__)

class OptimalPositionSizer:
    """
    Computes optimal fraction of bankroll to deploy utilizing the Kelly Extension
    bounded by Avellaneda-Stoikov inventory risk constraints.
    """
    def __init__(self, max_leverage: float = 1.0, half_life: float = 0.5, risk_aversion: float = 0.1):
        self.max_leverage = max_leverage
        # Kelly scaler (Half-Kelly is standard for risk reduction)
        self.kelly_fraction = half_life  
        # Gamma penalty for Avellaneda-Stoikov
        self.gamma = risk_aversion       
        
    def calculate_size(self, current_bankroll: float, win_prob: float, 
                      win_loss_ratio: float = 1.0, current_inventory: float = 0.0) -> float:
        """
        Calculates optimal USD size for entry.
        
        win_prob: Equivalent to the neural/Dempster-Shafer confidence (0 to 1)
        win_loss_ratio: Average win size / Average loss size
        current_inventory: Current total exposure in the market
        """
        # 1. Base Kelly Criterion
        # f* = p - (1 - p) / W
        if win_loss_ratio <= 0:
            win_loss_ratio = 1.0
            
        edge = win_prob - ((1.0 - win_prob) / win_loss_ratio)
        
        if edge <= 0:
            return 0.0
            
        optimal_f = edge * self.kelly_fraction
        
        # 2. Avellaneda-Stoikov Inventory Penalty
        # Q = current_inventory / bankroll (Normalized exposure)
        q = current_inventory / current_bankroll if current_bankroll > 0 else 0
        
        # The penalty scales exponentially with existing inventory utilizing gamma (risk aversion)
        # e.g., if we already hold 50% of bankroll in positions, we suppress new sizing
        inventory_penalty = math.exp(-self.gamma * abs(q) * 10.0)
        
        # Final adjusted fraction
        adjusted_f = optimal_f * inventory_penalty
        
        # Hard limits
        adjusted_f = min(self.max_leverage, adjusted_f)
        
        # Convert to USD
        optimal_usd = current_bankroll * adjusted_f
        
        logger.debug(
            f"KCE Sizing: Prob={win_prob:.2f} | Edge={edge:.3f} | "
            f"InvPen={inventory_penalty:.3f} | Final_frac={adjusted_f:.3f} | USD=${optimal_usd:.2f}"
        )
        
        return optimal_usd
