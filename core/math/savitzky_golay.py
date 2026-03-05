"""
Savitzky-Golay Convolutions & Price Kinematics (2026 SOTA)
Extracts ultra-precise momentum tensors (1st derivative) and acceleration (2nd derivative)
from raw tick data, mathematically filtering out microstructure jitter.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class SavitzkyGolayKinematics:
    """
    Computes smoothed derivatives using pre-calculated Savitzky-Golay convolution matrices
    for a fixed window size and polynomial order.
    """
    def __init__(self, window_size: int = 15, poly_order: int = 3):
        self.window_size = window_size
        self.poly_order = poly_order
        
        # Ensure window size is odd
        if self.window_size % 2 == 0:
            self.window_size += 1
            
        self._precompute_coefficients()
        
    def _precompute_coefficients(self):
        """Precompute the exact SG projection matrices."""
        # Half-window
        hw = self.window_size // 2
        # Define the x basis [-hw, ..., 0, ..., hw]
        x = np.arange(-hw, hw + 1, dtype=float)
        
        # Build the Design Matrix (Vandermonde)
        X = np.vstack([x**j for j in range(self.poly_order + 1)]).T
        
        # Moore-Penrose pseudoinverse to solve the least squares
        # coefficients matrix = (X^T * X)^-1 * X^T
        try:
            self.C = np.linalg.pinv(X.T @ X) @ X.T
        except Exception as e:
            logger.error(f"Failed to compute SG matrix: {e}")
            self.C = None

        # C[0] gives the smoothed value (0th derivative)
        # C[1] gives the 1st derivative (velocity)
        # C[2] gives the 2nd derivative (acceleration) * 2
        
    def get_kinematics(self, tick_series: list[float]) -> tuple[float, float, float]:
        """
        Extract exact (Smoothed Price, Velocity, Acceleration) for the most recent point.
        tick_series must be length == self.window_size.
        Returns: (price, velocity, acceleration)
        """
        if self.C is None or len(tick_series) < self.window_size:
            return 0.0, 0.0, 0.0
            
        # We only need the convolution at the *end* of the sequence to predict current state
        # Because standard SG targets the center of the window, an asymmetric endpoint SG
        # filter is mathematically optimal for real-time HFT forecasting without look-ahead bias.
        
        # Basic centered approach extrapolated to the Edge:
        y = np.array(tick_series[-self.window_size:])
        
        # To find the derivatives at the most recent point (x = hw)
        hw = self.window_size // 2
        
        # Polynomial coefficients: a_0, a_1, a_2...
        coeffs = self.C @ y
        
        # Evaluate polynomial at x = hw (the newest point)
        # P(x) = a0 + a1*x + a2*x^2 + a3*x^3
        # P'(x) = a1 + 2*a2*x + 3*a3*x^2
        # P''(x) = 2*a2 + 6*a3*x
        
        smoothed_price = sum(coeffs[j] * (hw**j) for j in range(self.poly_order + 1))
        
        velocity = sum(j * coeffs[j] * (hw**(j-1)) for j in range(1, self.poly_order + 1))
        
        acceleration = sum(j * (j-1) * coeffs[j] * (hw**(j-2)) for j in range(2, self.poly_order + 1))
        
        return smoothed_price, velocity, acceleration
