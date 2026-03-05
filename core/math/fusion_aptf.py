"""
Amortized Predictability-aware Training Framework (APTF) / Evidence Fusion
Replaces linear weighted summation with Dempster-Shafer Theory of Evidence,
allowing the engine to understand and quantify uncertainty when signals conflict.
"""

from typing import List, Dict

class DempsterShaferFusion:
    """
    Combines basic probability assignments (masses) from multiple signals.
    State Space: {Bullish}, {Bearish}, {Uncertainty ($\Theta$)}
    """
    def __init__(self):
        # We start with complete vacuous belief (100% Uncertainty)
        self.mass = {"bullish": 0.0, "bearish": 0.0, "uncertain": 1.0}
        
    def add_evidence(self, bullish_prob: float, bearish_prob: float, confidence: float, weight: float = 1.0):
        """
        Incorporate a new signal's evidence.
        confidence maps directly to the certainty of this evidence.
        """
        # Normalize incoming evidence
        adj_conf = confidence * weight
        if adj_conf >= 1.0:
            adj_conf = 0.999
            
        # Create mass vector for the incoming evidence
        m_in = {
            "bullish": bullish_prob * adj_conf,
            "bearish": bearish_prob * adj_conf,
            "uncertain": 1.0 - adj_conf
        }
        
        # Dempster's Rule of Combination
        # Intersection of sets. The conflict (K) measures contradictory evidence
        # (e.g. signal 1 says fully bullish, signal 2 says fully bearish).
        m_old = self.mass.copy()
        
        # Cross products
        k = (m_old["bullish"] * m_in["bearish"]) + (m_old["bearish"] * m_in["bullish"])
        
        if k == 1.0:
            norm = 1e-9 # avoid division by zero on absolute conflict
        else:
            norm = 1.0 - k
            
        m_new = {
            # Bullish = (Bullish AND Bullish) + (Bullish AND Uncertain) + (Uncertain AND Bullish)
            "bullish": (m_old["bullish"] * m_in["bullish"] + 
                        m_old["bullish"] * m_in["uncertain"] + 
                        m_old["uncertain"] * m_in["bullish"]) / norm,
                        
            # Bearish = (Bearish AND Bearish) + (Bearish AND Uncertain) + (Uncertain AND Bearish)
            "bearish": (m_old["bearish"] * m_in["bearish"] + 
                        m_old["bearish"] * m_in["uncertain"] + 
                        m_old["uncertain"] * m_in["bearish"]) / norm,
                        
            # Uncertain = (Uncertain AND Uncertain)
            "uncertain": (m_old["uncertain"] * m_in["uncertain"]) / norm
        }
        
        self.mass = m_new

    def get_consensus(self) -> tuple[float, float, float]:
        """Returns (Bullish Mass, Bearish Mass, Uncertainty)"""
        return self.mass["bullish"], self.mass["bearish"], self.mass["uncertain"]

    def get_decision(self, execution_threshold: float = 0.6) -> tuple[int, float]:
        """
        Returns (+1 for Bull, -1 for Bear, 0 for None) and the conviction score.
        Conviction is penalised heavily by remaining uncertainty.
        """
        bull = self.mass["bullish"]
        bear = self.mass["bearish"]
        unc = self.mass["uncertain"]
        
        if bull > bear and bull > execution_threshold:
            # We discount the conviction by uncertainty
            conviction = bull * (1.0 - unc)
            return 1, conviction
            
        if bear > bull and bear > execution_threshold:
            conviction = bear * (1.0 - unc)
            return -1, conviction
            
        return 0, 0.0
