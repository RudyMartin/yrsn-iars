"""
ApprovalRouter: Temperature-Aware Intelligent Routing

Implements the three-stream routing architecture (Green/Yellow/Red)
with temperature-quality duality from YRSN.

Key Innovation:
- τ = 1/α (temperature = 1/quality)
- High quality data → Low temperature → Tight rules → More automation
- Low quality data → High temperature → Loose rules → More human review
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime
import numpy as np

from ..adapters.cleanlab_adapter import CleanlabAdapter, YRSNResult, CollapseType
from ..adapters.temperature import (
    TemperatureScheduler, TemperatureConfig, TemperatureMode,
    AdaptiveRoutingEngine
)


class Stream(Enum):
    """Routing streams."""
    GREEN = "green"    # Auto-approve/reject
    YELLOW = "yellow"  # AI-assisted human review
    RED = "red"        # Expert manual review


class UrgencyLevel(Enum):
    """Urgency levels for queue prioritization."""
    STANDARD = "standard"
    EXPEDITED = "expedited"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class UrgencyWeights:
    """
    Weights for the formal urgency formula from IARS spec.

    U_total = (w₁ × T_expiry) + (w₂ × P_business) + (w₃ × S_sentiment) + (w₄ × C_clarity)

    Where:
    - T_expiry: Time to deadline (normalized 0-1)
    - P_business: Business impact/priority (normalized 0-1)
    - S_sentiment: NLP distress detection (normalized 0-1)
    - C_clarity: YRSN context clarity = (R - N) normalized to [0, 1]
    """
    w_expiry: float = 0.4       # Time to deadline weight
    w_business: float = 0.3     # Business impact weight
    w_sentiment: float = 0.2    # NLP distress detection weight
    w_clarity: float = 0.1      # YRSN context clarity weight


@dataclass
class RoutingDecision:
    """Complete routing decision with all metadata."""

    # Core decision
    request_id: str
    stream: Stream
    confidence_score: float

    # YRSN decomposition
    R: float
    S: float
    N: float
    collapse_type: CollapseType

    # Temperature information
    temperature: float
    quality_alpha: float

    # Urgency
    urgency: UrgencyLevel
    urgency_score: float

    # Thresholds used
    thresholds: Dict[str, float]

    # Soft routing (probabilities)
    p_green: float
    p_yellow: float
    p_red: float

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    knockout_rule: Optional[str] = None
    override_applied: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "request_id": self.request_id,
            "stream": self.stream.value,
            "confidence_score": self.confidence_score,
            "R": self.R,
            "S": self.S,
            "N": self.N,
            "collapse_type": self.collapse_type.value,
            "temperature": self.temperature,
            "quality_alpha": self.quality_alpha,
            "urgency": self.urgency.value,
            "urgency_score": self.urgency_score,
            "thresholds": self.thresholds,
            "p_green": self.p_green,
            "p_yellow": self.p_yellow,
            "p_red": self.p_red,
            "timestamp": self.timestamp.isoformat(),
            "knockout_rule": self.knockout_rule,
            "override_applied": self.override_applied,
            "reason": self.reason
        }


@dataclass
class ApprovalRequest:
    """Incoming approval request."""
    request_id: str
    text: str
    category: str
    amount: float
    requestor_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional enrichment
    budget_available: Optional[float] = None
    vendor_id: Optional[str] = None
    deadline: Optional[datetime] = None
    requestor_level: Optional[int] = None


class ApprovalRouter:
    """
    Intelligent Approval Router with Temperature-Quality Duality.

    Routes approval requests to appropriate streams based on:
    - Classifier confidence
    - YRSN context quality (R/S/N decomposition)
    - Temperature-adjusted thresholds
    - Urgency scoring
    - Knockout rules

    The key insight is τ = 1/α:
    - When data quality is high (α = R close to 1), temperature is low,
      thresholds are strict, and more requests go to green stream (automation)
    - When data quality is low (α = R close to 0), temperature is high,
      thresholds are loose, and more requests go to yellow/red (human review)
    """

    def __init__(
        self,
        cleanlab_adapter: Optional[CleanlabAdapter] = None,
        temperature_config: Optional[TemperatureConfig] = None,
        knockout_rules: Optional[List[Dict]] = None,
        base_thresholds: Optional[Dict[str, float]] = None
    ):
        self.adapter = cleanlab_adapter or CleanlabAdapter()
        self.engine = AdaptiveRoutingEngine(
            base_thresholds=base_thresholds,
            temperature_config=temperature_config
        )
        self.knockout_rules = knockout_rules or self._default_knockout_rules()

        # Tracking
        self._decisions: List[RoutingDecision] = []

    def route(
        self,
        request: ApprovalRequest,
        classifier_confidence: float,
        label_quality: float,
        pred_probs: Optional[np.ndarray] = None,
        ood_score: Optional[float] = None,
        is_duplicate: bool = False,
        force_temperature_mode: Optional[TemperatureMode] = None
    ) -> RoutingDecision:
        """
        Route an approval request to appropriate stream.

        Parameters
        ----------
        request : ApprovalRequest
            The incoming approval request
        classifier_confidence : float
            Classifier's prediction confidence [0, 1]
        label_quality : float
            Training data quality for this category [0, 1]
        pred_probs : np.ndarray, optional
            Full prediction probabilities for entropy calculation
        ood_score : float, optional
            Out-of-distribution score [0, 1]
        is_duplicate : bool
            Whether request is similar to recent requests
        force_temperature_mode : TemperatureMode, optional
            Force specific temperature mode (e.g., for shadow mode)

        Returns
        -------
        RoutingDecision
            Complete routing decision with metadata
        """
        # Step 1: Check knockout rules first
        knockout = self._check_knockout_rules(request)
        if knockout:
            return self._create_knockout_decision(request, knockout)

        # Step 2: Compute YRSN decomposition
        normalized_entropy = None
        normalized_margin = None

        if pred_probs is not None:
            entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10))
            max_entropy = np.log(len(pred_probs))
            normalized_entropy = entropy / max_entropy

            sorted_probs = np.sort(pred_probs)
            margin = sorted_probs[-1] - sorted_probs[-2]
            normalized_margin = (margin + 1) / 2

        yrsn = self.adapter.example_to_yrsn(
            label_quality=label_quality,
            normalized_margin=normalized_margin,
            normalized_entropy=normalized_entropy,
            ood_score=ood_score,
            is_duplicate=is_duplicate
        )

        # Step 3: Compute confidence score (combines classifier + YRSN)
        Cs = self.adapter.compute_confidence_score(
            yrsn=yrsn,
            classifier_confidence=classifier_confidence,
            label_quality=label_quality,
            ood_score=ood_score
        )

        # Step 4: Route with temperature-adjusted thresholds
        routing = self.engine.route(
            confidence_score=Cs,
            yrsn=yrsn,
            force_mode=force_temperature_mode
        )

        # Step 5: Compute urgency
        urgency, urgency_score = self._compute_urgency(request, yrsn)

        # Step 6: Build decision
        decision = RoutingDecision(
            request_id=request.request_id,
            stream=Stream(routing["stream"]),
            confidence_score=Cs,
            R=yrsn.R,
            S=yrsn.S,
            N=yrsn.N,
            collapse_type=yrsn.collapse_type,
            temperature=routing["temperature"],
            quality_alpha=routing["quality_alpha"],
            urgency=urgency,
            urgency_score=urgency_score,
            thresholds=routing["thresholds_used"],
            p_green=routing["soft_probabilities"]["green"],
            p_yellow=routing["soft_probabilities"]["yellow"],
            p_red=routing["soft_probabilities"]["red"],
            reason=routing.get("collapse_override")
        )

        self._decisions.append(decision)
        return decision

    def _check_knockout_rules(
        self,
        request: ApprovalRequest
    ) -> Optional[Dict[str, Any]]:
        """Check knockout rules (hard rules that bypass ML)."""
        for rule in self.knockout_rules:
            try:
                if rule["condition"](request):
                    return rule
            except Exception:
                continue
        return None

    def _create_knockout_decision(
        self,
        request: ApprovalRequest,
        knockout: Dict[str, Any]
    ) -> RoutingDecision:
        """Create decision for knockout rule match."""
        action = knockout.get("action", "REJECT")

        if action == "APPROVE":
            stream = Stream.GREEN
        elif action == "ROUTE_TO_EXPERT":
            stream = Stream.RED
        else:
            stream = Stream.RED  # Default to expert review for rejects

        return RoutingDecision(
            request_id=request.request_id,
            stream=stream,
            confidence_score=1.0,  # Knockout rules are definitive
            R=0.0, S=0.0, N=1.0,   # Mark as noise (rule-based, not ML)
            collapse_type=CollapseType.NONE,
            temperature=0.0,       # No temperature adjustment for rules
            quality_alpha=1.0,     # Rule confidence is 100%
            urgency=UrgencyLevel.CRITICAL,  # Knockouts are urgent
            urgency_score=1.0,
            thresholds={},
            p_green=1.0 if action == "APPROVE" else 0.0,
            p_yellow=0.0,
            p_red=0.0 if action == "APPROVE" else 1.0,
            knockout_rule=knockout.get("rule"),
            reason=knockout.get("message")
        )

    def _compute_urgency(
        self,
        request: ApprovalRequest,
        yrsn: YRSNResult,
        sentiment_score: Optional[float] = None,
        weights: Optional[UrgencyWeights] = None
    ) -> tuple[UrgencyLevel, float]:
        """
        Compute urgency using the formal weighted formula from IARS spec.

        U_total = (w₁ × T_expiry) + (w₂ × P_business) + (w₃ × S_sentiment) + (w₄ × C_clarity)

        Parameters
        ----------
        request : ApprovalRequest
            The incoming request with deadline, amount, etc.
        yrsn : YRSNResult
            YRSN decomposition for context clarity
        sentiment_score : float, optional
            NLP-detected distress/urgency in request text [0, 1]
        weights : UrgencyWeights, optional
            Custom weights for the formula components

        Returns
        -------
        tuple[UrgencyLevel, float]
            Urgency level enum and raw score [0, 1]
        """
        w = weights or UrgencyWeights()

        # === T_expiry: Time to deadline (normalized) ===
        T_expiry = 0.3  # Default: moderate urgency if no deadline
        if request.deadline:
            hours_remaining = (request.deadline - datetime.now()).total_seconds() / 3600
            if hours_remaining <= 0:
                T_expiry = 1.0  # Overdue
            elif hours_remaining < 2:
                T_expiry = 0.95
            elif hours_remaining < 8:
                T_expiry = 0.8
            elif hours_remaining < 24:
                T_expiry = 0.6
            elif hours_remaining < 72:
                T_expiry = 0.4
            else:
                T_expiry = 0.2

        # === P_business: Business impact (amount + requestor level) ===
        # Amount component (normalized to expected range)
        amount_score = 0.3  # Default
        if request.amount > 100000:
            amount_score = 1.0
        elif request.amount > 50000:
            amount_score = 0.8
        elif request.amount > 10000:
            amount_score = 0.6
        elif request.amount > 1000:
            amount_score = 0.4
        else:
            amount_score = 0.2

        # Requestor level component
        level_score = 0.3  # Default
        if request.requestor_level:
            if request.requestor_level >= 7:
                level_score = 1.0  # C-suite
            elif request.requestor_level >= 5:
                level_score = 0.7  # Director+
            elif request.requestor_level >= 3:
                level_score = 0.5  # Manager
            else:
                level_score = 0.3

        P_business = 0.6 * amount_score + 0.4 * level_score

        # === S_sentiment: NLP distress detection ===
        # If provided from external NLP, use it; otherwise default
        S_sentiment = sentiment_score if sentiment_score is not None else 0.3

        # === C_clarity: YRSN context clarity ===
        # (R - N) ranges from -1 to +1, normalize to [0, 1]
        # High R, low N = clear request = process faster
        # Low R, high N = unclear/noisy = needs clarification
        clarity_raw = yrsn.R - yrsn.N  # Range: [-1, 1]
        C_clarity = (clarity_raw + 1) / 2  # Normalize to [0, 1]

        # === Apply weighted formula ===
        urgency_score = (
            w.w_expiry * T_expiry +
            w.w_business * P_business +
            w.w_sentiment * S_sentiment +
            w.w_clarity * C_clarity
        )
        urgency_score = np.clip(urgency_score, 0.0, 1.0)

        # === Map to discrete level ===
        if urgency_score >= 0.8:
            level = UrgencyLevel.EMERGENCY
        elif urgency_score >= 0.6:
            level = UrgencyLevel.CRITICAL
        elif urgency_score >= 0.4:
            level = UrgencyLevel.EXPEDITED
        else:
            level = UrgencyLevel.STANDARD

        return level, urgency_score

    def _default_knockout_rules(self) -> List[Dict]:
        """Default knockout rules."""
        return [
            {
                "rule": "amount_exceeds_limit",
                "condition": lambda r: r.amount > 1_000_000,
                "message": "Amount exceeds $1M limit",
                "action": "ROUTE_TO_EXPERT"
            },
            {
                "rule": "budget_exhausted",
                "condition": lambda r: r.budget_available is not None and r.budget_available <= 0,
                "message": "Budget exhausted",
                "action": "REJECT"
            },
            {
                "rule": "below_auto_threshold",
                "condition": lambda r: r.amount < 100 and r.category == "supplies",
                "message": "Below auto-approval threshold",
                "action": "APPROVE"
            }
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        if not self._decisions:
            return {"n_decisions": 0}

        decisions = self._decisions
        n = len(decisions)

        # Stream distribution
        green_count = sum(1 for d in decisions if d.stream == Stream.GREEN)
        yellow_count = sum(1 for d in decisions if d.stream == Stream.YELLOW)
        red_count = sum(1 for d in decisions if d.stream == Stream.RED)

        # Temperature statistics
        temps = [d.temperature for d in decisions]
        alphas = [d.quality_alpha for d in decisions]

        return {
            "n_decisions": n,
            "automation_rate": green_count / n,
            "assisted_rate": yellow_count / n,
            "expert_rate": red_count / n,
            "avg_confidence": np.mean([d.confidence_score for d in decisions]),
            "avg_temperature": np.mean(temps),
            "avg_quality_alpha": np.mean(alphas),
            "temperature_std": np.std(temps),
            "knockout_rate": sum(1 for d in decisions if d.knockout_rule) / n,
            "collapse_rate": sum(1 for d in decisions if d.collapse_type != CollapseType.NONE) / n
        }

    def explain_temperature(self) -> str:
        """Explain temperature-quality relationship."""
        metrics = self.get_metrics()

        return f"""
TEMPERATURE-QUALITY ANALYSIS
============================

The YRSN framework uses τ = 1/α (temperature = 1/quality)

Current Statistics:
  Average Quality (α): {metrics.get('avg_quality_alpha', 0):.3f}
  Average Temperature (τ): {metrics.get('avg_temperature', 0):.3f}

Interpretation:
  - τ < 1.0: High quality data → Tight thresholds → More automation
  - τ = 1.0: Balanced quality → Standard thresholds
  - τ > 1.0: Low quality data → Loose thresholds → More human review

Stream Distribution:
  Green (Auto):    {metrics.get('automation_rate', 0)*100:.1f}%
  Yellow (Assist): {metrics.get('assisted_rate', 0)*100:.1f}%
  Red (Expert):    {metrics.get('expert_rate', 0)*100:.1f}%

To increase automation:
  1. Improve training data quality (raise α)
  2. Reduce noise in request context (lower N)
  3. Remove duplicates (lower S)
"""
