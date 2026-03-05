"""
Sigmoid & Logistic Confidence Bindings
Replaces brittle linear clamped piece-wise confidence functions with native
logistic curve scaling for continuous probability estimation.
"""
import math

def calculate_sigmoid_confidence(
    extremeness: float, 
    steepness: float = 6.0, 
    midpoint: float = 0.5, 
    max_confidence: float = 0.85,
    min_confidence_floor: float = 0.50
) -> float:
    """
    Logistic (Sigmoid) function to smoothly map a raw signal extremeness into
    a confidence percentage, avoiding harsh linear clamps and retaining continuous
    gradient resolution across the entire anomaly curve.
    
    extremeness: Raw intensity/distance metric (0.0 to 1.0+)
    steepness: How aggressively the confidence scales up (k)
    midpoint: The inflection point of the sigmoid (x0)
    """
    try:
        # Standard sigmoid: 1 / (1 + e^(-k * (x - x0)))
        sigmoid_val = 1.0 / (1.0 + math.exp(-steepness * (extremeness - midpoint)))
        
        # Scale to our output bounded range
        # E.g., if sigmoid_val is between 0 and 1, output is between floor and max
        bounded_conf = min_confidence_floor + (sigmoid_val * (max_confidence - min_confidence_floor))
        
        return min(max_confidence, max(min_confidence_floor, bounded_conf))
    except OverflowError:
        return min_confidence_floor if extremeness < midpoint else max_confidence
