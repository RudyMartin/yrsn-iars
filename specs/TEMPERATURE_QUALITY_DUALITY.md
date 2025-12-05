# Temperature-Quality Duality in YRSN-IARS

## The Core Insight: τ = 1/α

From Pedro Domingos' work and the YRSN research framework, temperature and quality are inversely related:

```
τ = 1/α

Where:
  τ (tau) = temperature - controls "tightness" of rules
  α (alpha) = quality = R / (R + S + N) = R (since R+S+N=1)
```

### Intuition

| Scenario | Quality (α) | Temperature (τ) | Behavior |
|----------|-------------|-----------------|----------|
| High-quality data | α → 1.0 | τ → 1.0 | Tight rules, confident decisions, more automation |
| Low-quality data | α → 0.0 | τ → ∞ | Loose rules, uncertain decisions, more human review |
| Balanced | α = 0.5 | τ = 2.0 | Moderate flexibility |

---

## Why Temperature Matters for Approval Routing

### The Problem with Fixed Thresholds

Traditional routing uses fixed thresholds:
```python
if confidence >= 0.95:
    route_to_green()
elif confidence >= 0.70:
    route_to_yellow()
else:
    route_to_red()
```

**Issue**: These thresholds assume uniform data quality. But in reality:
- Some categories have clean training data (high α)
- Some categories have noisy training data (low α)
- Quality varies over time as new data arrives

### The Temperature Solution

Adjust thresholds dynamically based on measured quality:

```python
# Compute quality from YRSN
α = R  # R is the relevant fraction

# Compute temperature
τ = 1 / α  # Clamped to [0.1, 3.0]

# Adjust thresholds
if τ < 1.0:  # High quality → tighter thresholds
    green_threshold = 0.95 + 0.02 * (1 - τ)  # Up to 0.97
    yellow_threshold = 0.70 + 0.02 * (1 - τ)  # Up to 0.72
else:  # Low quality → looser thresholds
    green_threshold = 0.95 - 0.05 * (τ - 1)  # Down to 0.85
    yellow_threshold = 0.70 - 0.05 * (τ - 1)  # Down to 0.60
```

---

## Temperature Modes

### 1. ADAPTIVE (Default)

Temperature computed from measured YRSN quality:

```python
from yrsn_iars.adapters.temperature import TemperatureMode

engine = AdaptiveRoutingEngine(
    temperature_config=TemperatureConfig(mode=TemperatureMode.ADAPTIVE)
)

# τ automatically adjusts based on each request's R value
decision = engine.route(confidence_score=0.85, yrsn=yrsn)
```

**Use when**: Normal operation, data quality varies

### 2. FIXED_TIGHT (τ = 0.1)

Very strict thresholds, minimal automation:

```python
decision = engine.route(
    confidence_score=0.85,
    yrsn=yrsn,
    force_mode=TemperatureMode.FIXED_TIGHT
)
```

**Use when**:
- Critical/sensitive decisions
- Regulatory requirements
- After system errors
- Manual audit mode

### 3. FIXED_LOOSE (τ = 2.0)

Flexible thresholds, maximum automation:

```python
decision = engine.route(
    confidence_score=0.85,
    yrsn=yrsn,
    force_mode=TemperatureMode.FIXED_LOOSE
)
```

**Use when**:
- Shadow mode (gathering data)
- Low-risk categories
- High-volume periods
- Stress testing

### 4. FIXED_MID (τ = 1.0)

Balanced mode (current default without temperature):

```python
decision = engine.route(
    confidence_score=0.85,
    yrsn=yrsn,
    force_mode=TemperatureMode.FIXED_MID
)
```

**Use when**:
- Comparing to non-temperature baseline
- A/B testing
- Initial deployment

### 5. ANNEALING

Temperature decreases over time (learning curve):

```python
config = TemperatureConfig(
    mode=TemperatureMode.ANNEALING,
    tau_initial=2.0,    # Start loose
    tau_final=0.3,      # End tight
    annealing_steps=1000
)
```

**Use when**:
- Initial system deployment
- After major model updates
- Training new categories

---

## Soft Routing with Temperature

Instead of hard routing decisions, temperature enables **soft probabilities**:

```python
from yrsn_iars.adapters.temperature import TemperatureScheduler

scheduler = TemperatureScheduler()
probs = scheduler.get_routing_probabilities(
    confidence_score=0.85,
    yrsn=YRSNResult(R=0.7, S=0.2, N=0.1)
)

# Low temperature (high quality):
# {"green": 0.75, "yellow": 0.20, "red": 0.05}  ← Sharp distribution

# High temperature (low quality):
# {"green": 0.40, "yellow": 0.35, "red": 0.25}  ← Soft distribution
```

### Softmax with Temperature

The probability distribution is computed using temperature-scaled softmax:

```
P(stream_i) = exp(score_i / τ) / Σ exp(score_j / τ)
```

| Temperature | Distribution | Interpretation |
|-------------|--------------|----------------|
| τ = 0.1 | Very sharp (winner-take-all) | Confident routing |
| τ = 1.0 | Standard softmax | Balanced |
| τ = 2.0 | Very soft (uniform-ish) | Uncertain routing |

---

## Setting Temperature: Guidelines

### Rule of Thumb

```python
# If you know your data quality:
if label_quality_avg > 0.8:
    mode = TemperatureMode.ADAPTIVE  # Trust the system
elif label_quality_avg > 0.6:
    mode = TemperatureMode.FIXED_MID  # Be cautious
else:
    mode = TemperatureMode.FIXED_LOOSE  # Gather more data
```

### By Use Case

| Use Case | Recommended Mode | Rationale |
|----------|------------------|-----------|
| Production (mature) | ADAPTIVE | Let quality drive decisions |
| Production (new) | ANNEALING | Start loose, tighten over time |
| Shadow mode | FIXED_LOOSE | Gather diverse examples |
| Audit mode | FIXED_TIGHT | Maximum human oversight |
| A/B testing | FIXED_MID | Consistent baseline |
| Critical decisions | FIXED_TIGHT | Safety over efficiency |
| High volume | FIXED_LOOSE | Throughput priority |

### By Category

Different approval categories may warrant different temperatures:

```python
CATEGORY_TEMPERATURE = {
    "supplies": TemperatureMode.FIXED_LOOSE,      # Low risk
    "software_license": TemperatureMode.ADAPTIVE,  # Medium risk
    "vendor_payment": TemperatureMode.ADAPTIVE,    # Medium risk
    "budget_variance": TemperatureMode.FIXED_TIGHT, # High risk
    "regulatory": TemperatureMode.FIXED_TIGHT,     # High risk
}
```

---

## Mathematical Foundation

### From YRSN Research

The temperature-quality duality comes from the phase transition analysis in YRSN:

```
α_ℓ = (ℓ - 1) / (2ℓ)  for ℓ-level decomposition
```

At the critical point α = 0.5 (where τ = 2), the system transitions between:
- **Ordered phase** (α > 0.5): Signal dominates, automation safe
- **Disordered phase** (α < 0.5): Noise dominates, human review needed

### Connection to Simulated Annealing

Temperature in YRSN follows the same intuition as simulated annealing:

- **High temperature**: System explores broadly, accepts many solutions
- **Low temperature**: System exploits narrowly, accepts only good solutions

For approval routing:
- High τ → Accept more borderline cases for automation
- Low τ → Only accept clear-cut cases for automation

---

## Implementation Details

### Temperature Bounds

```python
TAU_MIN = 0.1   # Minimum temperature (strictest rules)
TAU_MAX = 3.0   # Maximum temperature (loosest rules)

def compute_tau(R: float) -> float:
    if R < 0.01:
        return TAU_MAX
    tau = 1.0 / R
    return np.clip(tau, TAU_MIN, TAU_MAX)
```

### Threshold Adjustment Formula

```python
def adjust_threshold(base: float, tau: float, direction: str) -> float:
    """
    Adjust threshold based on temperature.

    direction="up": high τ raises threshold (stricter for that stream)
    direction="down": high τ lowers threshold (looser for that stream)
    """
    adjustment = 0.05 * (tau - 1.0)

    if direction == "up":
        return np.clip(base + adjustment, 0.5, 0.99)
    else:
        return np.clip(base - adjustment, 0.5, 0.99)
```

### Monitoring Temperature

```python
# Track temperature over time
scheduler = TemperatureScheduler()

# After many decisions...
stats = scheduler.get_statistics()
print(f"Avg α: {stats['alpha_mean']:.3f}")
print(f"Avg τ: {stats['tau_mean']:.3f}")
print(f"Correlation: {stats['correlation']:.3f}")  # Should be ~ -1.0
```

---

## Example: Full Routing with Temperature

```python
from yrsn_iars.adapters.cleanlab_adapter import CleanlabAdapter, YRSNResult
from yrsn_iars.adapters.temperature import (
    TemperatureConfig, TemperatureMode, AdaptiveRoutingEngine
)
from yrsn_iars.pipelines.approval_router import ApprovalRouter, ApprovalRequest

# Configure temperature
config = TemperatureConfig(
    mode=TemperatureMode.ADAPTIVE,
    tau_min=0.1,
    tau_max=3.0
)

# Create router
router = ApprovalRouter(temperature_config=config)

# Create request
request = ApprovalRequest(
    request_id="REQ-001",
    text="Request for AWS license renewal - $5,000",
    category="software_license",
    amount=5000,
    requestor_id="user@company.com"
)

# Route with Cleanlab quality signals
decision = router.route(
    request=request,
    classifier_confidence=0.88,
    label_quality=0.75,        # From Cleanlab
    ood_score=0.92,            # From Cleanlab
    is_duplicate=False
)

# Examine decision
print(f"Stream: {decision.stream.value}")
print(f"Confidence: {decision.confidence_score:.3f}")
print(f"Temperature: {decision.temperature:.3f}")
print(f"Quality (α): {decision.quality_alpha:.3f}")
print(f"YRSN: R={decision.R:.2f}, S={decision.S:.2f}, N={decision.N:.2f}")
print(f"Soft Probs: G={decision.p_green:.2f}, Y={decision.p_yellow:.2f}, R={decision.p_red:.2f}")
```

---

## Summary

**Temperature (τ) is the key lever for balancing automation vs. human oversight.**

- **τ = 1/α**: Temperature inversely proportional to quality
- **High α → Low τ**: Trust the ML, automate more
- **Low α → High τ**: Don't trust the ML, review more
- **Adaptive mode**: Let measured quality drive the balance
- **Fixed modes**: Override for specific use cases (audit, shadow, critical)

The elegance of this approach is that it's **self-calibrating**: as you improve data quality (via Cleanlab), the system automatically trusts itself more and routes more to automation.

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: YRSN-IARS Team*
