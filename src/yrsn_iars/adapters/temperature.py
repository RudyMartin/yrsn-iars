"""
Temperature-Quality Duality for YRSN-IARS

From Pedro Domingos' insight and yrsn-research: τ = 1/α

Temperature controls the "tightness" of routing rules:
- Low τ (high quality α): Tight rules, strict thresholds, high confidence required
- High τ (low quality α): Loose rules, flexible thresholds, accept uncertainty

This enables adaptive routing that becomes more conservative when
data quality is uncertain, and more aggressive when quality is high.

Key Concepts:
- α (alpha) = quality = R / (R + S + N) = R (since R+S+N=1)
- τ (tau) = temperature = 1/α
- Softmax with temperature: P(route_i) = exp(score_i/τ) / Σexp(score_j/τ)

Use Cases:
- Shadow mode: High τ (loose) to gather diverse examples
- Production mode: Dynamic τ based on data quality
- Critical decisions: Force low τ (tight) regardless of measured quality
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np

from .cleanlab_adapter import YRSNResult, CollapseType


class TemperatureMode(Enum):
    """Temperature control modes."""
    ADAPTIVE = "adaptive"      # τ = 1/α, dynamic based on quality
    FIXED_TIGHT = "tight"      # τ = 0.1, strict rules
    FIXED_LOOSE = "loose"      # τ = 2.0, flexible rules
    FIXED_MID = "mid"          # τ = 1.0, balanced (current default)
    ANNEALING = "annealing"    # τ decreases over time (learning)


@dataclass
class TemperatureConfig:
    """Configuration for temperature-based routing."""

    mode: TemperatureMode = TemperatureMode.ADAPTIVE

    # Bounds for adaptive mode
    tau_min: float = 0.1       # Minimum temperature (tightest rules)
    tau_max: float = 3.0       # Maximum temperature (loosest rules)

    # Fixed values for non-adaptive modes
    tau_tight: float = 0.1
    tau_mid: float = 1.0
    tau_loose: float = 2.0

    # Annealing schedule (for ANNEALING mode)
    tau_initial: float = 2.0   # Start loose
    tau_final: float = 0.3     # End tight
    annealing_steps: int = 1000

    # Quality thresholds (used to interpret α)
    alpha_low: float = 0.4     # Below this = high τ
    alpha_high: float = 0.7    # Above this = low τ


class TemperatureScheduler:
    """
    Manages temperature for routing decisions.

    Implements the core YRSN insight: τ = 1/α (temperature-quality duality)

    When quality is high (high α), we can be strict (low τ).
    When quality is low (low α), we should be flexible (high τ).
    """

    def __init__(self, config: Optional[TemperatureConfig] = None):
        self.config = config or TemperatureConfig()
        self.step = 0
        self._history: List[Tuple[float, float]] = []  # (alpha, tau) pairs

    def compute_tau(
        self,
        yrsn: YRSNResult,
        override_mode: Optional[TemperatureMode] = None
    ) -> float:
        """
        Compute temperature from YRSN quality.

        Parameters
        ----------
        yrsn : YRSNResult
            YRSN decomposition (R, S, N)
        override_mode : TemperatureMode, optional
            Override the configured mode

        Returns
        -------
        float
            Temperature τ for softmax/threshold adjustment
        """
        mode = override_mode or self.config.mode

        if mode == TemperatureMode.FIXED_TIGHT:
            return self.config.tau_tight

        if mode == TemperatureMode.FIXED_MID:
            return self.config.tau_mid

        if mode == TemperatureMode.FIXED_LOOSE:
            return self.config.tau_loose

        if mode == TemperatureMode.ANNEALING:
            return self._annealing_tau()

        # ADAPTIVE mode: τ = 1/α with bounds
        alpha = yrsn.R  # Quality = R (relevant fraction)

        # Avoid division by zero
        if alpha < 0.01:
            tau = self.config.tau_max
        else:
            tau = 1.0 / alpha

        # Clamp to bounds
        tau = np.clip(tau, self.config.tau_min, self.config.tau_max)

        # Record for analysis
        self._history.append((alpha, tau))
        self.step += 1

        return tau

    def _annealing_tau(self) -> float:
        """Compute annealing temperature (decreases over steps)."""
        progress = min(1.0, self.step / self.config.annealing_steps)

        # Linear annealing (can extend to cosine, exponential, etc.)
        tau = self.config.tau_initial + progress * (
            self.config.tau_final - self.config.tau_initial
        )

        self.step += 1
        return tau

    def adjust_thresholds(
        self,
        base_thresholds: Dict[str, float],
        tau: float
    ) -> Dict[str, float]:
        """
        Adjust routing thresholds based on temperature.

        High τ → looser thresholds (more automation)
        Low τ → tighter thresholds (more human review)

        Parameters
        ----------
        base_thresholds : dict
            Base threshold values, e.g., {"green": 0.95, "yellow": 0.70}
        tau : float
            Temperature

        Returns
        -------
        dict
            Adjusted thresholds
        """
        adjusted = {}

        for name, value in base_thresholds.items():
            if name == "green":
                # Green stream: higher τ → lower threshold (more automation)
                # At τ=1: use base value
                # At τ=2: allow 5% lower
                # At τ=0.5: require 5% higher
                adjustment = 0.05 * (1 - tau)
                adjusted[name] = np.clip(value + adjustment, 0.85, 0.99)

            elif name == "yellow":
                # Yellow stream: adjust lower bound
                adjustment = 0.05 * (1 - tau)
                adjusted[name] = np.clip(value + adjustment, 0.50, 0.85)

            else:
                # Other thresholds: scale by τ
                adjusted[name] = value

        return adjusted

    def softmax_with_temperature(
        self,
        scores: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """
        Apply softmax with temperature scaling.

        Low τ → sharp distribution (confident routing)
        High τ → soft distribution (uncertain routing)

        Parameters
        ----------
        scores : np.ndarray
            Raw scores for each routing option
        tau : float
            Temperature

        Returns
        -------
        np.ndarray
            Probability distribution over routing options
        """
        # Scale by temperature
        scaled = scores / tau

        # Numerical stability
        scaled = scaled - np.max(scaled)

        # Softmax
        exp_scores = np.exp(scaled)
        return exp_scores / np.sum(exp_scores)

    def get_routing_probabilities(
        self,
        confidence_score: float,
        yrsn: YRSNResult,
        stream_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get probability distribution over routing streams.

        Parameters
        ----------
        confidence_score : float
            Overall confidence Cs
        yrsn : YRSNResult
            YRSN decomposition
        stream_scores : dict, optional
            Pre-computed scores for each stream

        Returns
        -------
        dict
            {"green": p_green, "yellow": p_yellow, "red": p_red}
        """
        tau = self.compute_tau(yrsn)

        if stream_scores is None:
            # Default scoring based on Cs and YRSN
            stream_scores = {
                "green": confidence_score * yrsn.R,
                "yellow": confidence_score * (1 - yrsn.N),
                "red": (1 - confidence_score) + yrsn.N
            }

        scores = np.array([
            stream_scores["green"],
            stream_scores["yellow"],
            stream_scores["red"]
        ])

        probs = self.softmax_with_temperature(scores, tau)

        return {
            "green": float(probs[0]),
            "yellow": float(probs[1]),
            "red": float(probs[2]),
            "temperature": tau,
            "quality_alpha": yrsn.R
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get temperature history statistics."""
        if not self._history:
            return {"n_samples": 0}

        alphas, taus = zip(*self._history)

        return {
            "n_samples": len(self._history),
            "alpha_mean": np.mean(alphas),
            "alpha_std": np.std(alphas),
            "tau_mean": np.mean(taus),
            "tau_std": np.std(taus),
            "correlation": np.corrcoef(alphas, taus)[0, 1] if len(self._history) > 1 else 0
        }


class AdaptiveRoutingEngine:
    """
    Full routing engine with temperature-aware decision making.

    Combines:
    - Cleanlab data quality signals
    - YRSN context decomposition
    - Temperature-quality duality (τ = 1/α)
    - Adaptive thresholds based on measured quality
    """

    def __init__(
        self,
        base_thresholds: Optional[Dict[str, float]] = None,
        temperature_config: Optional[TemperatureConfig] = None
    ):
        self.base_thresholds = base_thresholds or {
            "green": 0.95,   # Cs threshold for auto-approve
            "yellow": 0.70,  # Cs threshold for assisted review
            "min_R": 0.5,    # Minimum R for green stream
            "max_N": 0.3,    # Maximum N allowed
        }
        self.scheduler = TemperatureScheduler(temperature_config)

    def route(
        self,
        confidence_score: float,
        yrsn: YRSNResult,
        force_mode: Optional[TemperatureMode] = None
    ) -> Dict[str, Any]:
        """
        Make routing decision with temperature-adjusted thresholds.

        Parameters
        ----------
        confidence_score : float
            Overall confidence Cs [0, 1]
        yrsn : YRSNResult
            YRSN decomposition
        force_mode : TemperatureMode, optional
            Force specific temperature mode

        Returns
        -------
        dict
            Routing decision with metadata
        """
        # Compute temperature
        tau = self.scheduler.compute_tau(yrsn, force_mode)

        # Adjust thresholds based on temperature
        thresholds = self.scheduler.adjust_thresholds(self.base_thresholds, tau)

        # Get soft routing probabilities
        probs = self.scheduler.get_routing_probabilities(
            confidence_score, yrsn
        )

        # Make hard routing decision
        if (confidence_score >= thresholds["green"] and
            yrsn.R >= thresholds.get("min_R", 0.5) and
            yrsn.N <= thresholds.get("max_N", 0.3)):
            stream = "green"
        elif confidence_score >= thresholds["yellow"]:
            stream = "yellow"
        else:
            stream = "red"

        # Check for collapse conditions (force red)
        if yrsn.collapse_type in [CollapseType.CONFUSION, CollapseType.POISONING]:
            stream = "red"
            reason = f"Collapse detected: {yrsn.collapse_type.value}"
        else:
            reason = None

        return {
            "stream": stream,
            "confidence_score": confidence_score,
            "temperature": tau,
            "quality_alpha": yrsn.R,
            "thresholds_used": thresholds,
            "soft_probabilities": probs,
            "yrsn": yrsn.to_dict(),
            "collapse_override": reason
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_temperature(R: float, tau_min: float = 0.1, tau_max: float = 3.0) -> float:
    """
    Compute temperature from quality (R component).

    τ = 1/α where α = R (relevant fraction)

    Parameters
    ----------
    R : float
        Relevant fraction from YRSN [0, 1]
    tau_min, tau_max : float
        Bounds for temperature

    Returns
    -------
    float
        Temperature τ
    """
    if R < 0.01:
        return tau_max
    tau = 1.0 / R
    return np.clip(tau, tau_min, tau_max)


def adjust_threshold_by_temperature(
    base_threshold: float,
    tau: float,
    direction: str = "up"
) -> float:
    """
    Adjust a single threshold by temperature.

    Parameters
    ----------
    base_threshold : float
        Base threshold value
    tau : float
        Temperature
    direction : str
        "up" = high τ raises threshold (stricter)
        "down" = high τ lowers threshold (looser)

    Returns
    -------
    float
        Adjusted threshold
    """
    # Adjustment magnitude scales with τ deviation from 1.0
    adjustment = 0.05 * (tau - 1.0)

    if direction == "up":
        return np.clip(base_threshold + adjustment, 0.5, 0.99)
    else:
        return np.clip(base_threshold - adjustment, 0.5, 0.99)


# =============================================================================
# Example Usage Documentation
# =============================================================================

"""
TEMPERATURE USAGE EXAMPLES
==========================

1. Basic Temperature Calculation:

    from yrsn_iars.adapters.temperature import compute_temperature

    # High quality data → low temperature → tight rules
    tau = compute_temperature(R=0.8)  # tau ≈ 1.25

    # Low quality data → high temperature → loose rules
    tau = compute_temperature(R=0.3)  # tau ≈ 3.0 (capped)

2. Adaptive Routing:

    from yrsn_iars.adapters.temperature import AdaptiveRoutingEngine

    engine = AdaptiveRoutingEngine()

    decision = engine.route(
        confidence_score=0.85,
        yrsn=YRSNResult(R=0.7, S=0.2, N=0.1)
    )

    print(f"Stream: {decision['stream']}")
    print(f"Temperature: {decision['temperature']}")

3. Shadow Mode (High Temperature):

    from yrsn_iars.adapters.temperature import TemperatureMode

    decision = engine.route(
        confidence_score=0.85,
        yrsn=yrsn,
        force_mode=TemperatureMode.FIXED_LOOSE  # τ = 2.0
    )

4. Critical Decision (Low Temperature):

    decision = engine.route(
        confidence_score=0.85,
        yrsn=yrsn,
        force_mode=TemperatureMode.FIXED_TIGHT  # τ = 0.1
    )

5. Soft Routing Probabilities:

    probs = scheduler.get_routing_probabilities(
        confidence_score=0.85,
        yrsn=yrsn
    )

    # Returns: {"green": 0.6, "yellow": 0.3, "red": 0.1, "temperature": 1.4}
    # Low τ → sharper distribution
    # High τ → softer distribution
"""
