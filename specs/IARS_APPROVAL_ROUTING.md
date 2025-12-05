# IARS: Intelligent Approval Routing System

## Cleanlab + YRSN Integration for Approval Classification

This document specifies how Cleanlab's data quality signals and YRSN's context decomposition power the **Intelligence Layer** of an Intelligent Approval Routing System.

---

## I. The IARS Vision

Transform approval pipelines from reactive FIFO queues into **proactive, data-driven decision engines** using:

1. **Cleanlab** - Detect noisy/mislabeled training data for the classifier
2. **YRSN** - Decompose request context quality (R/S/N) for routing confidence
3. **AWS Bedrock** - LLM-powered intent classification and urgency detection

### The Three-Stream Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              INCOMING REQUESTS                   │
                    │  (Email, Slack, Jira, ERPs, APIs)               │
                    └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │         INGESTION & ENRICHMENT LAYER            │
                    │  • Unified intake from all sources              │
                    │  • Zero-touch enrichment (budget, history)      │
                    │  • Context extraction                           │
                    └─────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTELLIGENCE LAYER (Cleanlab + YRSN)                  │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  INTENT          │  │  CONFIDENCE      │  │  YRSN CONTEXT            │  │
│  │  CLASSIFICATION  │  │  SCORING (Cs)    │  │  DECOMPOSITION           │  │
│  │                  │  │                  │  │                          │  │
│  │  • Software Lic. │  │  Cleanlab:       │  │  R = Relevant info       │  │
│  │  • Budget Var.   │  │  • label_quality │  │  S = Superfluous detail  │  │
│  │  • Travel Req.   │  │  • ood_score     │  │  N = Noise/ambiguity     │  │
│  │  • PTO Request   │  │  • data_shapley  │  │                          │  │
│  │  • Vendor Appr.  │  │                  │  │  Collapse Detection:     │  │
│  │                  │  │  Combined into:  │  │  • POISONING (bad data)  │  │
│  │                  │  │  Cs = f(scores)  │  │  • CONFUSION (unclear)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    URGENCY SCORING (U_total)                          │  │
│  │                                                                        │  │
│  │  U = (w₁ × T_expiry) + (w₂ × P_business) + (w₃ × S_sentiment)        │  │
│  │                                                                        │  │
│  │  T_expiry   = Time to deadline (normalized)                           │  │
│  │  P_business = Business impact (C-suite, production blocker)           │  │
│  │  S_sentiment= NLP distress/escalation detection                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│    GREEN STREAM         │ │    YELLOW STREAM        │ │    RED STREAM           │
│    AUTO-RESOLUTION      │ │    ASSISTED REVIEW      │ │    EXPERT TRIAGE        │
│                         │ │                         │ │                         │
│  Criteria:              │ │  Criteria:              │ │  Criteria:              │
│  • Cs > 95%             │ │  • 70% < Cs < 95%       │ │  • Cs < 70%             │
│  • Low Risk             │ │  • Medium Risk          │ │  • High Risk            │
│  • R > 0.7, N < 0.1     │ │  • 0.4 < R < 0.7        │ │  • R < 0.4 OR N > 0.3   │
│                         │ │                         │ │                         │
│  Action:                │ │  Action:                │ │  Action:                │
│  AUTO-APPROVE/REJECT    │ │  AI + Human Review      │ │  Manual Expert Review   │
│                         │ │  (suggestion + reason)  │ │  (domain specialist)    │
│                         │ │                         │ │                         │
│  SLA: < 5 seconds       │ │  SLA: < 4 hours         │ │  SLA: < 24 hours        │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

---

## II. Cleanlab's Role: Training Data Quality

### Problem
The approval classifier is trained on historical approval data, but:
- **Mislabeled examples**: Past approvals/rejections may have been wrong
- **Inconsistent labeling**: Different approvers apply different standards
- **Ambiguous cases**: Some requests genuinely unclear

### Solution: Cleanlab Quality Audit

```python
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from cleanlab.multiannotator import get_label_quality_multiannotator

# Historical approval decisions from multiple approvers
historical_decisions = pd.read_csv("approval_history.csv")

# Train classifier on historical data
clf = train_approval_classifier(historical_decisions)
pred_probs = cross_val_predict(clf, features, decisions, method='predict_proba')

# Find problematic training examples
label_quality = get_label_quality_scores(decisions, pred_probs)
label_issues = find_label_issues(decisions, pred_probs)

# For multi-approver data
if multiple_approvers:
    annotator_quality = get_label_quality_multiannotator(
        labels_multiannotator=approver_decisions,
        pred_probs=pred_probs
    )
    # Identify unreliable approvers
    low_quality_approvers = annotator_quality[
        annotator_quality['annotator_quality'] < 0.7
    ]
```

### Cleanlab Signals for IARS

| Signal | Use in IARS | Impact |
|--------|-------------|--------|
| `label_quality` | Training data curation | Remove/relabel bad examples |
| `is_label_issue` | Flag inconsistent decisions | Retrain without noise |
| `annotator_agreement` | Approver consistency | Identify training needs |
| `ood_score` | Novel request detection | Route to human review |
| `consensus_label` | Ground truth estimation | Better training labels |

---

## III. YRSN's Role: Request Context Quality

### The R/S/N Decomposition for Approvals

Every approval request has context (description, attachments, metadata). YRSN decomposes this into:

| Component | Definition | Example |
|-----------|------------|---------|
| **R (Relevant)** | Information needed for decision | "Budget: $5,000 for AWS services" |
| **S (Superfluous)** | True but unnecessary detail | "The team discussed this in our Monday meeting" |
| **N (Noise)** | Misleading/incorrect/ambiguous | Conflicting amounts, unclear purpose |

### Context Quality → Routing Decision

```python
from yrsn_context import detect_collapse, CollapseType

def assess_request_context(request_text: str, enriched_data: dict) -> dict:
    """
    Analyze request context quality for routing decision.
    """
    # Extract YRSN components
    R, S, N = decompose_request_context(request_text, enriched_data)

    # Detect collapse type
    analysis = detect_collapse(R, S, N)

    # Determine routing implications
    routing = {
        "context_quality": R + 0.5 * S,  # Y-score
        "context_risk": S + 1.5 * N,     # Risk score
        "collapse_type": analysis.collapse_type,
        "requires_clarification": analysis.collapse_type == CollapseType.CONFUSION,
        "missing_information": analysis.collapse_type == CollapseType.POISONING,
    }

    return routing
```

### YRSN Collapse Types in Approval Context

| Collapse Type | Approval Meaning | Action |
|---------------|------------------|--------|
| **NONE** | Clear, complete request | Route normally |
| **POISONING** | Missing/incorrect key info | Request clarification |
| **DISTRACTION** | Too much irrelevant detail | Summarize for reviewer |
| **CONFUSION** | Contradictory information | Expert triage required |
| **CLASH** | Conflicting approver opinions | Escalate to senior |

---

## IV. The Confidence Score (Cs) Formula

The confidence score combines Cleanlab signals with YRSN decomposition:

```python
def compute_confidence_score(
    # Cleanlab signals
    label_quality: float,      # From training data quality
    ood_score: float,          # Is this request typical?
    classifier_confidence: float,  # Model's prediction confidence

    # YRSN signals
    R: float,                  # Relevant context ratio
    S: float,                  # Superfluous context ratio
    N: float,                  # Noise ratio

    # Weights (tunable)
    w_label: float = 0.2,
    w_ood: float = 0.2,
    w_classifier: float = 0.3,
    w_context: float = 0.3
) -> float:
    """
    Compute overall confidence score for routing.

    Returns Cs in [0, 1] where:
    - Cs > 0.95: Green Stream (auto-approve)
    - 0.70 < Cs < 0.95: Yellow Stream (assisted)
    - Cs < 0.70: Red Stream (expert)
    """
    # Cleanlab contribution: penalize if training data was noisy
    cleanlab_score = (label_quality + ood_score) / 2

    # YRSN contribution: context quality
    context_score = R + 0.3 * S  # Penalize noise, tolerate some superfluous

    # Combine
    Cs = (
        w_label * label_quality +
        w_ood * ood_score +
        w_classifier * classifier_confidence +
        w_context * context_score
    )

    return min(1.0, max(0.0, Cs))
```

### Routing Thresholds

| Stream | Cs Range | YRSN Criteria | Action |
|--------|----------|---------------|--------|
| **Green** | > 0.95 | R > 0.7, N < 0.1 | Auto-approve/reject |
| **Yellow** | 0.70 - 0.95 | 0.4 < R < 0.7 | AI suggestion + human |
| **Red** | < 0.70 | R < 0.4 OR N > 0.3 | Expert manual review |

---

## V. Urgency Scoring with YRSN Enhancement

### Base Urgency Formula

```
U_total = (w₁ × T_expiry) + (w₂ × P_business) + (w₃ × S_sentiment)
```

### YRSN-Enhanced Urgency

```python
def compute_enhanced_urgency(
    # Base urgency factors
    time_to_deadline: float,    # Normalized [0, 1], 1 = imminent
    business_impact: float,     # [0, 1], 1 = critical
    sentiment_urgency: float,   # NLP-detected distress [0, 1]

    # YRSN factors
    R: float,                   # High R = clear request = faster processing
    N: float,                   # High N = needs clarification = deprioritize

    # Weights
    w1: float = 0.4,  # Time weight
    w2: float = 0.3,  # Business weight
    w3: float = 0.2,  # Sentiment weight
    w4: float = 0.1   # Context clarity bonus
) -> float:
    """
    Compute urgency score with YRSN context bonus.

    Clear requests (high R) get urgency boost.
    Noisy requests (high N) get urgency penalty (need clarification first).
    """
    base_urgency = (
        w1 * time_to_deadline +
        w2 * business_impact +
        w3 * sentiment_urgency
    )

    # Context clarity modifier
    # Clear requests can be processed faster → boost urgency
    # Noisy requests need clarification first → reduce urgency
    clarity_modifier = (R - N)  # Ranges from -1 to +1

    enhanced_urgency = base_urgency + (w4 * clarity_modifier)

    return min(1.0, max(0.0, enhanced_urgency))
```

### Urgency Options (User-Selectable)

```python
URGENCY_OPTIONS = {
    "standard": {
        "description": "Normal processing queue",
        "sla_hours": 24,
        "auto_escalate_after": 48
    },
    "expedited": {
        "description": "Faster review, requires justification",
        "sla_hours": 8,
        "auto_escalate_after": 16,
        "requires_justification": True
    },
    "critical": {
        "description": "Immediate attention, senior review",
        "sla_hours": 2,
        "auto_escalate_after": 4,
        "requires_justification": True,
        "requires_manager_approval": True
    },
    "emergency": {
        "description": "Business-critical, executive escalation",
        "sla_hours": 0.5,
        "auto_escalate_after": 1,
        "requires_justification": True,
        "requires_executive_notification": True
    }
}
```

---

## VI. Override Mechanism

### Override Types

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class OverrideType(Enum):
    """Types of routing overrides."""
    FORCE_GREEN = "force_auto_approve"
    FORCE_YELLOW = "force_assisted_review"
    FORCE_RED = "force_expert_triage"
    CHANGE_URGENCY = "change_urgency_level"
    REASSIGN = "reassign_approver"
    BYPASS_RULES = "bypass_business_rules"

class OverrideAuthority(Enum):
    """Who can authorize overrides."""
    SYSTEM = "system"           # Automated rules
    APPROVER = "approver"       # Standard approver
    MANAGER = "manager"         # Manager level
    EXECUTIVE = "executive"     # C-suite
    COMPLIANCE = "compliance"   # Compliance officer


@dataclass
class OverrideRequest:
    """Request to override routing decision."""
    request_id: str
    override_type: OverrideType
    requested_by: str
    authority_level: OverrideAuthority
    justification: str
    original_routing: str
    new_routing: str
    timestamp: datetime
    expires_at: Optional[datetime] = None


@dataclass
class OverridePolicy:
    """Policy for override authorization."""
    override_type: OverrideType
    min_authority: OverrideAuthority
    requires_justification: bool
    requires_second_approval: bool
    audit_required: bool
    max_duration_hours: Optional[int] = None


# Override Policies
OVERRIDE_POLICIES = {
    OverrideType.FORCE_GREEN: OverridePolicy(
        override_type=OverrideType.FORCE_GREEN,
        min_authority=OverrideAuthority.MANAGER,
        requires_justification=True,
        requires_second_approval=True,  # Dual control
        audit_required=True,
        max_duration_hours=None  # One-time
    ),
    OverrideType.FORCE_RED: OverridePolicy(
        override_type=OverrideType.FORCE_RED,
        min_authority=OverrideAuthority.APPROVER,
        requires_justification=True,
        requires_second_approval=False,
        audit_required=True,
        max_duration_hours=None
    ),
    OverrideType.CHANGE_URGENCY: OverridePolicy(
        override_type=OverrideType.CHANGE_URGENCY,
        min_authority=OverrideAuthority.APPROVER,
        requires_justification=True,
        requires_second_approval=False,
        audit_required=True,
        max_duration_hours=None
    ),
    OverrideType.BYPASS_RULES: OverridePolicy(
        override_type=OverrideType.BYPASS_RULES,
        min_authority=OverrideAuthority.EXECUTIVE,
        requires_justification=True,
        requires_second_approval=True,
        audit_required=True,
        max_duration_hours=24  # Temporary only
    )
}
```

### Override Processing

```python
class OverrideManager:
    """Manages routing overrides with audit trail."""

    def __init__(self, audit_logger, notification_service):
        self.audit_logger = audit_logger
        self.notifications = notification_service

    def request_override(
        self,
        override_request: OverrideRequest
    ) -> dict:
        """
        Process an override request.

        Returns approval status and any required actions.
        """
        policy = OVERRIDE_POLICIES.get(override_request.override_type)

        if policy is None:
            return {"approved": False, "reason": "Unknown override type"}

        # Check authority level
        if not self._has_authority(
            override_request.authority_level,
            policy.min_authority
        ):
            return {
                "approved": False,
                "reason": f"Insufficient authority. Requires: {policy.min_authority.value}"
            }

        # Check justification
        if policy.requires_justification and not override_request.justification:
            return {
                "approved": False,
                "reason": "Justification required for this override type"
            }

        # Check if second approval needed
        if policy.requires_second_approval:
            return {
                "approved": False,
                "pending": True,
                "reason": "Requires second approval",
                "action": "route_to_second_approver"
            }

        # Approve and audit
        self._apply_override(override_request)
        self._audit_override(override_request, policy)

        return {
            "approved": True,
            "effective_until": override_request.expires_at,
            "audit_id": self.audit_logger.last_id
        }

    def _has_authority(
        self,
        user_authority: OverrideAuthority,
        required_authority: OverrideAuthority
    ) -> bool:
        """Check if user has sufficient authority."""
        authority_hierarchy = [
            OverrideAuthority.SYSTEM,
            OverrideAuthority.APPROVER,
            OverrideAuthority.MANAGER,
            OverrideAuthority.EXECUTIVE,
            OverrideAuthority.COMPLIANCE
        ]
        user_level = authority_hierarchy.index(user_authority)
        required_level = authority_hierarchy.index(required_authority)
        return user_level >= required_level

    def _apply_override(self, override_request: OverrideRequest):
        """Apply the override to the request routing."""
        # Implementation depends on routing system
        pass

    def _audit_override(
        self,
        override_request: OverrideRequest,
        policy: OverridePolicy
    ):
        """Log override for compliance audit."""
        self.audit_logger.log({
            "event": "ROUTING_OVERRIDE",
            "request_id": override_request.request_id,
            "override_type": override_request.override_type.value,
            "requested_by": override_request.requested_by,
            "authority": override_request.authority_level.value,
            "justification": override_request.justification,
            "original_routing": override_request.original_routing,
            "new_routing": override_request.new_routing,
            "timestamp": override_request.timestamp.isoformat(),
            "policy_version": "1.0"
        })
```

---

## VII. Knockout Criteria (Hard Rules)

Knockout criteria trigger **immediate automatic actions** regardless of confidence scores:

```python
KNOCKOUT_CRITERIA = {
    # Immediate Rejection (no human review needed)
    "auto_reject": [
        {
            "rule": "amount_exceeds_absolute_limit",
            "condition": lambda r: r.amount > 1_000_000,
            "message": "Amount exceeds $1M absolute limit",
            "action": "REJECT",
            "notification": ["compliance", "executive"]
        },
        {
            "rule": "blocked_vendor",
            "condition": lambda r: r.vendor_id in BLOCKED_VENDORS,
            "message": "Vendor is on blocked list",
            "action": "REJECT",
            "notification": ["compliance"]
        },
        {
            "rule": "duplicate_request",
            "condition": lambda r: r.is_duplicate and r.duplicate_age_hours < 24,
            "message": "Duplicate of recent request",
            "action": "REJECT",
            "notification": ["requestor"]
        },
        {
            "rule": "budget_exhausted",
            "condition": lambda r: r.available_budget <= 0,
            "message": "Budget fully exhausted",
            "action": "REJECT",
            "notification": ["requestor", "manager"]
        }
    ],

    # Immediate Approval (pre-authorized)
    "auto_approve": [
        {
            "rule": "pre_approved_subscription",
            "condition": lambda r: r.item_id in PRE_APPROVED_ITEMS,
            "message": "Item is pre-approved",
            "action": "APPROVE",
            "notification": []
        },
        {
            "rule": "below_auto_threshold",
            "condition": lambda r: r.amount < 100 and r.category == "supplies",
            "message": "Below auto-approval threshold",
            "action": "APPROVE",
            "notification": []
        }
    ],

    # Force Expert Review (high risk)
    "force_expert": [
        {
            "rule": "regulatory_sensitive",
            "condition": lambda r: r.category in REGULATED_CATEGORIES,
            "message": "Regulatory review required",
            "action": "ROUTE_TO_COMPLIANCE",
            "notification": ["compliance"]
        },
        {
            "rule": "executive_requestor",
            "condition": lambda r: r.requestor_level >= EXECUTIVE_LEVEL,
            "message": "Executive request - special handling",
            "action": "ROUTE_TO_SENIOR",
            "notification": ["senior_approver"]
        },
        {
            "rule": "first_time_vendor",
            "condition": lambda r: r.vendor_id not in KNOWN_VENDORS,
            "message": "New vendor requires vetting",
            "action": "ROUTE_TO_PROCUREMENT",
            "notification": ["procurement"]
        }
    ]
}


def apply_knockout_rules(request) -> Optional[dict]:
    """
    Check knockout criteria before confidence scoring.

    Returns action dict if knockout triggered, None otherwise.
    """
    for category, rules in KNOCKOUT_CRITERIA.items():
        for rule in rules:
            if rule["condition"](request):
                return {
                    "knockout": True,
                    "category": category,
                    "rule": rule["rule"],
                    "message": rule["message"],
                    "action": rule["action"],
                    "notifications": rule["notification"]
                }

    return None  # No knockout, proceed to confidence scoring
```

---

## VIII. Metrics & Dashboard

### Key Performance Indicators

```python
IARS_METRICS = {
    # Throughput Metrics
    "automation_rate": {
        "description": "% of requests handled by Green Stream",
        "target": "> 60%",
        "formula": "green_stream_count / total_requests"
    },
    "mean_time_to_resolution": {
        "description": "Average time from submission to decision",
        "target": {
            "green": "< 5 seconds",
            "yellow": "< 4 hours",
            "red": "< 24 hours"
        },
        "formula": "avg(decision_time - submission_time)"
    },

    # Quality Metrics
    "false_positive_rate": {
        "description": "% of auto-approvals later reversed",
        "target": "< 2%",
        "formula": "reversed_auto_approvals / total_auto_approvals"
    },
    "override_rate": {
        "description": "% of AI suggestions overridden by humans",
        "target": "< 15%",
        "formula": "overridden_suggestions / total_yellow_stream"
    },
    "yrsn_accuracy": {
        "description": "Correlation between YRSN scores and outcomes",
        "target": "> 0.8",
        "formula": "correlation(predicted_quality, actual_outcome)"
    },

    # Context Quality Metrics
    "avg_R_score": {
        "description": "Average relevance of request context",
        "target": "> 0.6",
        "formula": "mean(R_scores)"
    },
    "clarification_rate": {
        "description": "% of requests requiring clarification",
        "target": "< 20%",
        "formula": "clarification_requests / total_requests"
    },

    # Operational Metrics
    "queue_depth": {
        "description": "Pending requests per approver",
        "target": "< 10",
        "alert_threshold": 15
    },
    "escalation_rate": {
        "description": "% of requests escalated due to SLA breach",
        "target": "< 5%",
        "formula": "escalated_requests / total_requests"
    }
}
```

### Dashboard Components

```python
DASHBOARD_PANELS = {
    "real_time": [
        "current_queue_depth_by_stream",
        "requests_per_minute",
        "avg_confidence_score_last_hour",
        "active_overrides"
    ],
    "daily_summary": [
        "automation_rate_trend",
        "mttr_by_stream",
        "top_10_request_categories",
        "yrsn_distribution_histogram"
    ],
    "quality_analysis": [
        "false_positive_trend",
        "override_reasons_breakdown",
        "collapse_type_distribution",
        "model_retraining_triggers"
    ],
    "compliance": [
        "knockout_triggers_by_rule",
        "override_audit_log",
        "approver_consistency_scores",
        "regulatory_review_backlog"
    ]
}
```

---

## IX. Implementation Phases

### Phase 1: Shadow Mode (Weeks 1-4)

```python
PHASE_1_CONFIG = {
    "mode": "shadow",
    "actions": {
        "green_stream": "log_only",      # Log decision, don't execute
        "yellow_stream": "log_only",
        "red_stream": "log_only"
    },
    "calibration": {
        "collect_human_decisions": True,
        "compare_to_model": True,
        "adjust_thresholds": True
    },
    "deliverables": [
        "Baseline metrics report",
        "Threshold calibration document",
        "Edge case catalog"
    ]
}
```

### Phase 2: Low-Risk Automation (Weeks 5-8)

```python
PHASE_2_CONFIG = {
    "mode": "partial_live",
    "actions": {
        "green_stream": {
            "auto_approve": ["supplies", "software_renewal"],
            "auto_reject": ["blocked_vendor", "duplicate"],
            "log_only": ["everything_else"]
        },
        "yellow_stream": "ai_suggestion",
        "red_stream": "manual_only"
    },
    "safeguards": {
        "daily_audit": True,
        "rollback_trigger": "false_positive_rate > 5%",
        "human_spot_check_rate": 0.1  # 10% sample
    },
    "deliverables": [
        "Automation impact report",
        "Approver feedback summary",
        "Model performance metrics"
    ]
}
```

### Phase 3: Full Scale (Week 9+)

```python
PHASE_3_CONFIG = {
    "mode": "full_live",
    "actions": {
        "green_stream": "auto_execute",
        "yellow_stream": "ai_augmented",
        "red_stream": "expert_routing"
    },
    "features": {
        "dynamic_queue_reordering": True,
        "load_balancing": True,
        "continuous_model_retraining": True,
        "yrsn_collapse_alerts": True
    },
    "optimization": {
        "threshold_auto_tuning": True,
        "approver_capacity_modeling": True,
        "urgency_prediction": True
    }
}
```

---

## X. Cleanlab + YRSN Integration Points

### Training Pipeline

```
Historical Approvals → Cleanlab Analysis → Clean Training Data → Classifier
        ↓                    ↓                                       ↓
   Raw labels          label_quality                            pred_probs
   Multi-approver      annotator_agreement                           ↓
   Decisions           consensus_label                      Confidence Score
                                                                     ↓
                                                            Routing Decision
```

### Inference Pipeline

```
New Request → Context Extraction → YRSN Decomposition → Cs Calculation → Routing
     ↓               ↓                    ↓                   ↓            ↓
  Raw text      R, S, N ratios      Collapse type      Cs = f(...)    Stream
  Metadata      Context quality     POISONING?         [0, 1]         G/Y/R
  Enrichment    Risk score          CONFUSION?
```

### Feedback Loop

```
Human Decision → Compare to AI → Update Cleanlab → Retrain Model
       ↓               ↓               ↓               ↓
   Approve/Reject   Override?     Add to training    New classifier
   With reason      Rate >15%?    Mark quality       Updated thresholds
                    → Alert        Adjust weights
```

---

## Appendix: Quick Reference

### Routing Decision Matrix

| Cs | R | N | Collapse | Stream | Action |
|----|---|---|----------|--------|--------|
| >0.95 | >0.7 | <0.1 | NONE | GREEN | Auto |
| 0.70-0.95 | 0.4-0.7 | 0.1-0.3 | LOW | YELLOW | Assisted |
| <0.70 | <0.4 | >0.3 | ANY | RED | Expert |
| Any | Any | Any | CONFUSION | RED | Clarify |
| Any | Any | Any | POISONING | YELLOW | Verify |

### Override Authority Matrix

| Override Type | Approver | Manager | Executive | Compliance |
|---------------|----------|---------|-----------|------------|
| Force Green | - | ✓ (+2nd) | ✓ | ✓ |
| Force Yellow | ✓ | ✓ | ✓ | ✓ |
| Force Red | ✓ | ✓ | ✓ | ✓ |
| Change Urgency | ✓ | ✓ | ✓ | ✓ |
| Bypass Rules | - | - | ✓ (+2nd) | ✓ |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: YRSN-IARS Team*
