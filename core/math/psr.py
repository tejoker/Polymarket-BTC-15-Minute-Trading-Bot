"""
Probabilistic Sharpe Ratio (PSR)
Validates strategy statistical edge by discounting for non-normal
return distributions (skewness and kurtosis) over annualized high-frequency scales.
"""

import math
import numpy as np
import scipy.stats as st
import logging

logger = logging.getLogger(__name__)

class ProbabilisticSharpeRatio:
    """
    Computes Marcos Lopez de Prado's Probabilistic Sharpe Ratio (PSR).
    Determines if the observed Sharpe ratio is statistically significant.
    """
    def __init__(self, benchmark_sharpe: float = 0.0, periods_per_year: int = 35040):
        # 365 days * 24 hours * 4 intervals (15-min) = 35040
        self.benchmark_sr = benchmark_sharpe
        self.annualization_factor = math.sqrt(periods_per_year)
        
    def calculate_psr(self, returns: list[float]) -> tuple[float, float, float]:
        """
        Calculate annualized SR, PSR, and confidence.
        Returns: (Annualized Sharpe, PSR Probability, True/False if > 95% confident)
        """
        if len(returns) < 30:
            return 0.0, 0.0, False
            
        ret_array = np.array(returns)
        mean_ret = np.mean(ret_array)
        std_ret = np.std(ret_array)
        
        if std_ret == 0:
            return 0.0, 0.0, False
            
        obs_sr = mean_ret / std_ret
        ann_sr = obs_sr * self.annualization_factor
        
        # Calculate moments for non-normality adjustment
        skewness = st.skew(ret_array)
        # st.kurtosis computes excess kurtosis (Normal = 0). 
        # PSR formula uses non-excess, so we add 3.
        kurtosis = st.kurtosis(ret_array) + 3.0 
        
        n = len(ret_array)
        
        # Denominator of PSR Z-statistic
        # sqrt(1 - skew * SR + (kurtosis - 1) * SR^2 / 4)
        term1 = 1.0
        term2 = skewness * obs_sr
        term3 = ((kurtosis - 1.0) / 4.0) * (obs_sr ** 2)
        
        denominator_variance = term1 - term2 + term3
        if denominator_variance <= 0:
            denominator_variance = 0.0001
            
        psr_stat = ((obs_sr - (self.benchmark_sr / self.annualization_factor)) * math.sqrt(n - 1)) / math.sqrt(denominator_variance)
        
        # Cumulative Density Function of Standard Normal
        prob = st.norm.cdf(psr_stat)
        
        is_significant = prob >= 0.95
        
        logger.info(
            f"PSR Validation | n={n}, Ann.SR={ann_sr:.2f}, "
            f"Skew={skewness:.2f}, Kurt={kurtosis:.2f} | "
            f"Stat Sig Prob={prob:.2%} (Pass: {is_significant})"
        )
        
        return ann_sr, prob, is_significant
