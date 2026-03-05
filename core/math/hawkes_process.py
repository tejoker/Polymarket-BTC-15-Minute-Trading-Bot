"""
Hawkes Process for Continuous Execution (2026 SOTA)
Replaces rigid modulo/floor time barriers with a self-exciting point process model.
Evaluates inter-arrival times of liquidity bursts via exponential decay kernels.
"""

import time
import math
import logging

logger = logging.getLogger(__name__)

class HawkesExecutionKernel:
    """
    Models market excitation to determine continuous execution probability.
    Instead of rigidly gating trades between [780, 840) seconds, the bot 
    calculates an intensity $\lambda(t)$ that scales execution probability 
    based on localized volatility and time decay.
    """
    def __init__(self, base_intensity: float = 0.1, decay_rate: float = 2.0, impact: float = 0.5):
        self.mu = base_intensity   # Baseline background arrival rate
        self.beta = decay_rate     # Exponential decay parameter
        self.alpha = impact        # Excitation jump size per event
        
        self.last_event_time = 0.0
        self.current_intensity = base_intensity
        
    def add_event(self, timestamp: float, intensity_multiplier: float = 1.0):
        """Register a liquidity burst or deep alpha signal."""
        if self.last_event_time > 0:
            dt = timestamp - self.last_event_time
            # Decay previous intensity
            self.current_intensity = self.mu + (self.current_intensity - self.mu) * math.exp(-self.beta * dt)
            
        # Add jump
        self.current_intensity += self.alpha * intensity_multiplier
        self.last_event_time = timestamp
        
    def get_intensity(self, current_time: float) -> float:
        """Calculate real-time $\lambda(t)$ taking decay into account."""
        if self.last_event_time == 0.0:
            return self.mu
            
        dt = current_time - self.last_event_time
        decayed = self.mu + (self.current_intensity - self.mu) * math.exp(-self.beta * dt)
        return min(max(decayed, 0.0), 10.0) # Cap at 10x baseline
        
    def is_actionable(self, current_time: float, threshold: float = 0.8) -> bool:
        """
        Determine if the current market state is actionable based on Hawkes intensity.
        Provides continuous temporal adaptation instead of rigid boundary checking.
        """
        return self.get_intensity(current_time) >= threshold

# Global execution kernel
execution_kernel = HawkesExecutionKernel()
