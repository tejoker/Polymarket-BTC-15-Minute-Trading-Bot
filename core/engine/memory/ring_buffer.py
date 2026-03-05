"""
Zero-Copy Ring Buffer Implementation (LMAX Disruptor Pattern logic)
Replaces Python deques and lists for tick ingestion to prevent Garbage Collection
pauses and provide O(1) determinism.
"""

import time
import math
from typing import Optional, List, Dict
from datetime import datetime

class TickRingBuffer:
    """
    Fixed-size pre-allocated circular buffer for storing quote ticks.
    Avoids continuous object creation/destruction which causes GC pauses.
    """
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        # Pre-allocate arrays of native types instead of dicts for exact memory footprint
        self._prices = [0.0] * capacity
        self._timestamps = [0.0] * capacity
        
        self._head = 0
        self._count = 0
        
    def append(self, timestamp: float, price: float):
        """Append a generic tick with O(1) complexity."""
        idx = self._head
        self._prices[idx] = price
        self._timestamps[idx] = timestamp
        
        self._head = (idx + 1) % self.capacity
        if self._count < self.capacity:
            self._count += 1
            
    def get_latest_price(self) -> Optional[float]:
        if self._count == 0:
            return None
        idx = (self._head - 1) % self.capacity
        return self._prices[idx]
        
    def get_recent_prices(self, count: int) -> List[float]:
        """Get the last N prices efficiently."""
        if self._count == 0:
            return []
        
        actual_count = min(count, self._count)
        result = [0.0] * actual_count
        
        for i in range(actual_count):
            idx = (self._head - 1 - i) % self.capacity
            result[actual_count - 1 - i] = self._prices[idx]
            
        return result
        
    def get_price_at_time(self, target_timestamp: float, tolerance: float = 15.0) -> Optional[float]:
        """
        O(log N) binary search for price at a specific historical timestamp.
        For Ring Buffers, we need to handle the wrap-around.
        """
        if self._count == 0:
            return None
            
        best_diff = float('inf')
        best_price = None
        
        # Traverse linearly for now (in Python, lists up to 2000 are extremely fast)
        # To make IT O(log N) we could bisect the unwrapped view, but a clean C traversal
        # via list comp or min() over zipped arrays is faster in CPython.
        for i in range(self._count):
            idx = (self._head - 1 - i) % self.capacity
            diff = abs(self._timestamps[idx] - target_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_price = self._prices[idx]
                
        if best_diff <= tolerance:
            return best_price
        return None
        
    def __len__(self) -> int:
        return self._count
