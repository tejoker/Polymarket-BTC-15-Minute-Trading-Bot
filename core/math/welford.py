"""
Welford's Online Algorithm
Provides numerically stable, O(1) single-pass computation of variance and mean.
Replaces naive SMA calculations and population variance equations.
"""

import math

class WelfordRollingStat:
    """
    Maintains rolling Mean and Variance utilizing Welford's online algorithm
    adapted for a fixed lookback window.
    """
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._buffer = [0.0] * window_size
        self._head = 0
        self._count = 0
        
        # Welford state
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float):
        """Add a new value and remove the oldest value from the running stats."""
        if self._count < self.window_size:
            # Traditional Welford addition
            self._count += 1
            delta = value - self.mean
            self.mean += delta / self._count
            delta2 = value - self.mean
            self.M2 += delta * delta2
            
            # Store value
            self._buffer[self._head] = value
            self._head = (self._head + 1) % self.window_size
        else:
            # We must remove the old value and add the new one
            old_value = self._buffer[self._head]
            
            # Remove old
            delta = old_value - self.mean
            self.mean -= delta / self.window_size
            delta2 = old_value - self.mean
            self.M2 -= delta * delta2
            
            # Add new
            delta = value - self.mean
            self.mean += delta / self.window_size
            delta2 = value - self.mean
            self.M2 += delta * delta2
            
            # Store value
            self._buffer[self._head] = value
            self._head = (self._head + 1) % self.window_size
            
            # Numerical stability correction for float decay over millions of ticks
            if self.M2 < 0:
                self.M2 = 0.0

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self.M2 / (self._count - 1)  # Sample variance

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)
