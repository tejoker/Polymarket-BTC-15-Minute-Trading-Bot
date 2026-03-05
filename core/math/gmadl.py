"""
Generalized Mean Absolute Directional Loss (GMADL) — FTRL Gradient Adapter

MATHEMATICAL BACKGROUND:
  Standard MADL (Mean Absolute Directional Loss):
    L_MADL(ŷ, r) = |r| * sign(-ŷ * r)

  Problem: sign() is non-differentiable. Gradient vanishes or explodes at zero,
  making it incompatible with gradient-based optimizers.

  GMADL solution — replace sign() with a smooth parameterized sigmoid:
    L_GMADL(ŷ, r) = mean( |r|^p * sigmoid(-k * ŷ * r)^β )

  Where:
    r   = realized return (positive = asset went up)
    ŷ   = predicted directional score (positive = bullish, negative = bearish)
    k   = directional sensitivity: sharpness of the sigmoid partition at zero
    p   = magnitude exponent: how much to amplify large-move trade gradients
    β   = sigmoid power: further concentrates penalty on the wrong side

  Key properties:
    - Fully differentiable → compatible with gradient-based optimizers (including FTRL)
    - When signal is correct and move is large: loss ≈ 0 (well-predicted, no penalty)
    - When signal is wrong and move is large: loss ≈ |r|^p (strongly penalized)
    - When the move is tiny (|r| ≈ 0): loss ≈ 0 regardless (noise discarded)

APPLICATION IN THIS CODEBASE:
  GMADL is applied NOT as a neural-network training loss, but as a replacement
  for the flat win-rate / PnL gradient currently fed into FTRLProximal in
  feedback/learning_engine.py.

  Current gradient (flat):
    gradient = (0.5 - win_rate) * 0.6 + (-pnl / 100) * 0.4

  GMADL gradient (magnitude-weighted directional):
    r_t   = trade.pnl_pct for LONG trades, -trade.pnl_pct for SHORT trades
            (converts to BTC-frame: positive when BTC went up)
    ŷ_it  = direction_sign_i × avg_confidence_i
            (+1 × confidence if source i was BULLISH, -1 × confidence if BEARISH)
    g_i   = mean_t[ ∂L_GMADL/∂ŷ ] over all trades t involving source i

  Result: signals that were confidently right on large BTC moves get boosted,
  signals that were confidently wrong on large moves get suppressed, and the
  common near-coin-flip small-move trades contribute minimal gradient noise.

GRADIENT DERIVATION (β=1):
  L  = |r|^p * σ(-k * ŷ * r)
  ∂L/∂ŷ = |r|^p * ∂σ(-k*ŷ*r)/∂ŷ
         = |r|^p * σ(-k*ŷ*r) * (1 - σ(-k*ŷ*r)) * (-k*r)

  Interpretation:
    - σ(-k*ŷ*r) is large when signal is wrong → gradient is large → FTRL penalizes
    - σ(-k*ŷ*r) is small when signal is right → gradient ≈ 0 → weight preserved
    - (-k*r) encodes both sign and magnitude: large |r| amplifies the update
"""
import math
from typing import List, Tuple


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid that avoids exp overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def gmadl_loss(
    r: float,
    y_hat: float,
    k: float = 10.0,
    p: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute GMADL loss for a single (realized_return, predicted_score) pair.

    Args:
        r:    Realized return. Positive = asset went up, negative = went down.
              In BTC binary context: pnl_pct for LONG trades, -pnl_pct for SHORT.
        y_hat: Predicted directional score.
              Typically: direction_sign × confidence ∈ [-1, +1]
              +1 × conf = strong bullish prediction
              -1 × conf = strong bearish prediction
        k:    Directional sensitivity (default 10.0).
              Higher values → sharper sigmoid partition → stricter correctness check.
        p:    Magnitude exponent (default 1.0).
              Higher values → larger BTC moves contribute exponentially more to loss.
        beta: Sigmoid power (default 1.0 = standard sigmoid).

    Returns:
        Loss value ≥ 0. Near 0 = good prediction, large = bad prediction on big move.
    """
    magnitude = abs(r) ** p
    sig = _sigmoid(-k * y_hat * r)
    return magnitude * (sig ** beta)


def gmadl_gradient(
    r: float,
    y_hat: float,
    k: float = 10.0,
    p: float = 1.0,
) -> float:
    """
    Compute ∂L_GMADL/∂ŷ for β=1 (standard sigmoid).

    Used as the FTRL update gradient per signal source.

    Sign convention matches FTRLProximal.update(feature_id, loss_gradient):
      Negative gradient → loss is decreasing → signal is performing well → weight boosted
      Positive gradient → loss is increasing → signal is performing poorly → weight suppressed

    Args:
        r:     Realized return in BTC-frame (positive = BTC went up).
        y_hat: direction_sign × signal_confidence.
        k:     Directional sensitivity.
        p:     Magnitude exponent.

    Returns:
        Gradient scalar for FTRL update.
    """
    sig = _sigmoid(-k * y_hat * r)
    # ∂/∂ŷ [|r|^p * σ(-k*ŷ*r)] = |r|^p * σ * (1-σ) * (-k*r)
    magnitude = abs(r) ** p
    return magnitude * sig * (1.0 - sig) * (-k * r)


def compute_source_gradient(
    trades: List[Tuple[float, float]],
    k: float = 10.0,
    p: float = 1.0,
) -> float:
    """
    Compute the mean GMADL gradient across all trades for one signal source.

    This is the replacement for the flat win-rate / PnL gradient in the
    learning engine's calculate_optimal_weights method.

    Args:
        trades: List of (r_t, y_hat_t) pairs where:
                  r_t    = realized BTC-frame return for trade t
                  y_hat_t = direction_sign_i × avg_confidence_i for source i
        k:      Directional sensitivity (default 10.0).
        p:      Magnitude exponent (default 1.0).

    Returns:
        Mean gradient across trades. Negative = good performance (boost weight).
        Positive = poor performance (suppress weight).
        Returns 0.0 if trades list is empty.
    """
    if not trades:
        return 0.0

    gradients = [gmadl_gradient(r, yhat, k=k, p=p) for r, yhat in trades]
    return sum(gradients) / len(gradients)
