# Code Review: YRSN-IARS Integration

## Issues Found and Fixes Required

### 1. **Missing Weighted Urgency Score Formula**

**Issue**: The `_compute_urgency` method in `approval_router.py` uses additive logic but doesn't implement the formal weighted formula from the IARS spec:

```
U_total = (w₁ × T_expiry) + (w₂ × P_business) + (w₃ × S_sentiment)
```

**Fix Required**: Add proper weighted urgency with configurable weights and sentiment analysis.

---

### 2. **Missing Cleanlab Functions**

We're not using these important Cleanlab signals:

| Function | Purpose | YRSN Mapping |
|----------|---------|--------------|
| `get_confident_thresholds()` | Per-class decision boundaries | Identifies weak classes (S) |
| `rank_classes_by_label_quality()` | Class-level quality | Dataset-level S analysis |
| `find_overlapping_classes()` | Confused class pairs | Source of N (confusion) |
| `overall_label_health_score()` | Dataset health | Global R estimate |
| `get_label_quality_ensemble_scores()` | Multi-model consensus | More robust N detection |
| `get_majority_vote_label()` | Multi-annotator consensus | Ground truth for multi-approver |
| `OutOfDistribution.fit_score()` | OOD detection | Direct N signal |
| `data_shapley_knn()` | Data valuation | R/S distinction |
| `find_top_issues()` | Prioritized issues | Focus N detection |

---

### 3. **Batch Processing Inefficiency**

**Issue**: `batch_to_yrsn()` uses iterrows() which is slow for large DataFrames.

**Fix**: Use vectorized operations.

---

### 4. **Missing Annotator Quality Integration**

**Issue**: Multi-annotator signals not integrated for approval scenarios with multiple approvers.

**Fix**: Add `multiannotator_to_yrsn()` method.

---

### 5. **Temperature Edge Cases**

**Issue**: When R=0, τ becomes infinity. Current code handles this but could be clearer.

**Fix**: Add explicit handling and documentation.

---

### 6. **Missing Datalab Integration**

**Issue**: Cleanlab's `Datalab` class provides unified issue detection but isn't used.

**Fix**: Add `datalab_to_yrsn()` method.

---

## Missing Cleanlab Signals for Complete YRSN

### Noise (N) Signals
- `find_label_issues()` ✓ (implemented)
- `label_quality_score` ✓ (implemented)
- `ood_score` ✓ (implemented)
- `find_overlapping_classes()` ✗ (MISSING - class confusion → N)
- `get_confident_thresholds()` ✗ (MISSING - per-class reliability)
- `is_label_issue` ✓ (implemented)

### Superfluous (S) Signals
- `is_near_duplicate` ✓ (implemented)
- `normalized_entropy` ✓ (implemented)
- `near_duplicate_cluster_id` ✗ (MISSING - duplicate grouping)
- `rank_classes_by_label_quality()` ✗ (MISSING - class-level quality)

### Relevant (R) Signals
- `data_shapley_knn()` ✓ (partially implemented)
- `overall_label_health_score()` ✗ (MISSING - global R)
- `consensus_quality_score` ✗ (MISSING - multi-annotator)
- `annotator_agreement` ✗ (MISSING - inter-rater reliability)

### Dataset-Level
- `confident_joint` ✓ (implemented)
- `noise_matrix` ✗ (MISSING - per-class noise rates)
- `health_summary()` ✗ (MISSING - comprehensive audit)

---

## Notebook Use Case Matrix

| Notebook | Domain | Difficulty | Primary Cleanlab Functions | YRSN Focus | Collapse Type |
|----------|--------|------------|---------------------------|------------|---------------|
| 01 | Approval Data | Easy | `find_label_issues`, `get_label_quality_scores` | Basic R/S/N | POISONING |
| 02 | Text Classification | Easy | `filter`, `rank`, text quality | Text → YRSN | DISTRACTION |
| 03 | Multi-Annotator | Medium | `multiannotator`, consensus | Annotator quality | CLASH |
| 04 | RAG/Retrieval | Medium | `outlier`, `dataset` | Context quality | CONFUSION |
| 05 | Token/NER | Hard | `token_classification` | Sequence quality | POISONING |
| 06 | Production Pipeline | Hard | `Datalab`, AWS integration | Full pipeline | ALL |

---

## Required Code Changes

### 1. Enhanced CleanlabAdapter

```python
# Add these methods to CleanlabAdapter

def from_datalab(self, lab: "Datalab") -> pd.DataFrame:
    """Convert Datalab results to YRSN."""
    pass

def from_multiannotator(
    self,
    consensus_quality: np.ndarray,
    annotator_agreement: np.ndarray,
    num_annotations: np.ndarray
) -> YRSNResult:
    """Convert multi-annotator results to YRSN."""
    pass

def from_overlapping_classes(
    self,
    overlap_df: pd.DataFrame
) -> Dict[str, float]:
    """Get class-level confusion (N) from overlapping classes."""
    pass

def from_class_quality(
    self,
    class_quality_df: pd.DataFrame
) -> Dict[str, YRSNResult]:
    """Get per-class YRSN from class quality ranking."""
    pass
```

### 2. Enhanced Urgency Formula

```python
@dataclass
class UrgencyWeights:
    """Weights for urgency formula."""
    w_expiry: float = 0.4      # Time to deadline
    w_business: float = 0.3    # Business impact
    w_sentiment: float = 0.2   # NLP distress detection
    w_clarity: float = 0.1     # YRSN context clarity

def compute_weighted_urgency(
    self,
    request: ApprovalRequest,
    yrsn: YRSNResult,
    sentiment_score: Optional[float] = None,
    weights: Optional[UrgencyWeights] = None
) -> Tuple[UrgencyLevel, float]:
    """
    Compute urgency using formal weighted formula:

    U = w₁×T_expiry + w₂×P_business + w₃×S_sentiment + w₄×(R-N)
    """
    pass
```

### 3. Vectorized Batch Processing

```python
def batch_to_yrsn_vectorized(
    self,
    cleanlab_df: pd.DataFrame,
    ...
) -> pd.DataFrame:
    """Vectorized YRSN conversion for large datasets."""
    # Use numpy broadcasting instead of iterrows
    pass
```
