# YRSN-IARS Notebook Use Case Matrix

This document outlines the 6 progressive notebooks demonstrating Cleanlab + YRSN integration for the Intelligent Approval Routing System.

## Overview Matrix

| # | Notebook | Domain | Difficulty | Primary Cleanlab Functions | YRSN Focus | Collapse Type | AWS Services |
|---|----------|--------|------------|---------------------------|------------|---------------|--------------|
| 01 | Basic Approval Data | Corporate Approvals | Easy | `find_label_issues`, `get_label_quality_scores` | Basic R/S/N decomposition | POISONING | S3, Bedrock |
| 02 | Text Classification | Support Tickets | Easy | `filter`, `rank`, text quality | Text → YRSN mapping | DISTRACTION | Comprehend, S3 |
| 03 | Multi-Annotator | Committee Decisions | Medium | `multiannotator`, consensus | Approver agreement | CLASH | SageMaker |
| 04 | RAG/Retrieval | Policy Retrieval | Medium | `outlier`, `dataset`, OOD | Context quality | CONFUSION | Bedrock KB, Kendra |
| 05 | Token/NER | Contract Entities | Hard | `token_classification` | Sequence quality | POISONING | Comprehend |
| 06 | Production Pipeline | Full IARS | Hard | `Datalab`, all integrations | Complete pipeline | ALL | Full stack |

---

## Detailed Breakdown

### Notebook 01: Basic Approval Data Quality

**Domain**: Corporate purchase approvals (standard business scenario)

**Difficulty**: ⭐ Easy

**Learning Goals**:
- Understand basic Cleanlab label quality detection
- Map Cleanlab outputs to YRSN R/S/N components
- See temperature-quality duality in action

**Cleanlab Functions Used**:
```python
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from cleanlab.count import estimate_cv_predicted_probabilities
```

**YRSN Mapping**:
| Cleanlab Signal | → | YRSN Component |
|-----------------|---|----------------|
| `label_quality_score` | → | N (inverse) |
| `is_label_issue` | → | N = 1.0 |
| High confidence + quality | → | R (high) |

**Collapse Type**: POISONING (mislabeled approval decisions)

**Sample Data Structure**:
```python
{
    "request_id": "REQ-001",
    "text": "Purchase request for AWS licenses",
    "amount": 5000,
    "category": "software_license",
    "historical_label": "approved",  # May be wrong!
    "approver_id": "mgr_123"
}
```

**AWS Services**: S3 (data storage), Bedrock (embeddings)

---

### Notebook 02: Text Classification Quality

**Domain**: IT support ticket routing

**Difficulty**: ⭐ Easy

**Learning Goals**:
- Handle text-specific quality issues
- Detect toxic/PII/informal content
- Route ambiguous tickets correctly

**Cleanlab Functions Used**:
```python
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores, get_confidence_weighted_entropy_for_each_label
# Text-specific quality via Cleanlab Studio or custom
```

**YRSN Mapping**:
| Cleanlab Signal | → | YRSN Component |
|-----------------|---|----------------|
| Toxic content | → | N (harmful) |
| PII detected | → | N (compliance risk) |
| Informal language | → | S (not ideal) |
| Clear, professional | → | R (high) |

**Collapse Type**: DISTRACTION (verbose/unfocused tickets)

**Sample Data Structure**:
```python
{
    "ticket_id": "TKT-42",
    "subject": "URGENT!!!!! laptop broken",
    "body": "my laptop isnt working i need it fixed asap plz help!!!!!",
    "category": "hardware",  # Possibly wrong
    "priority": "high"
}
```

**AWS Services**: S3, Comprehend (sentiment/PII), Bedrock

---

### Notebook 03: Multi-Annotator Consensus

**Domain**: Committee approval decisions (multiple approvers)

**Difficulty**: ⭐⭐ Medium

**Learning Goals**:
- Handle multiple approvers with different opinions
- Find consensus among disagreeing annotators
- Identify problematic approvers (always yes/no)

**Cleanlab Functions Used**:
```python
from cleanlab.multiannotator import (
    get_label_quality_multiannotator,
    get_majority_vote_label,
    get_active_learning_scores
)
```

**YRSN Mapping**:
| Cleanlab Signal | → | YRSN Component |
|-----------------|---|----------------|
| `quality_of_consensus` (low) | → | N (unclear decision) |
| `annotator_agreement` (low) | → | S (contentious) |
| `quality_of_consensus` (high) | → | R (clear decision) |
| Specific annotator quality (low) | → | Flag annotator |

**Collapse Type**: CLASH (approvers disagree)

**Sample Data Structure**:
```python
{
    "request_id": "REQ-100",
    "approver_decisions": [
        {"approver": "mgr_A", "decision": "approve"},
        {"approver": "mgr_B", "decision": "reject"},
        {"approver": "dir_C", "decision": "approve"}
    ],
    "final_decision": "approve"  # 2-1 vote
}
```

**AWS Services**: SageMaker (custom model), DynamoDB (approver history)

---

### Notebook 04: RAG/Retrieval Context Quality

**Domain**: Policy document retrieval for approval decisions

**Difficulty**: ⭐⭐ Medium

**Learning Goals**:
- Detect out-of-distribution queries
- Evaluate retrieval context quality
- Handle irrelevant or outdated policy matches

**Cleanlab Functions Used**:
```python
from cleanlab.outlier import OutOfDistribution
from cleanlab.dataset import find_overlapping_classes
from cleanlab.rank import get_confidence_weighted_entropy_for_each_label
```

**YRSN Mapping**:
| Cleanlab Signal | → | YRSN Component |
|-----------------|---|----------------|
| `ood_score` (high) | → | N (novel query) |
| Class overlap | → | N (confusing categories) |
| Low retrieval confidence | → | S (ambiguous context) |
| High confidence + in-dist | → | R (good context) |

**Collapse Type**: CONFUSION (wrong policy retrieved)

**Sample Data Structure**:
```python
{
    "query": "Can I expense a team lunch?",
    "retrieved_policies": [
        {"doc": "expense_policy_v2.pdf", "chunk": "...", "score": 0.82},
        {"doc": "travel_policy.pdf", "chunk": "...", "score": 0.78}  # Wrong!
    ],
    "policy_version": "2024-Q1"
}
```

**AWS Services**: Bedrock Knowledge Base, Kendra, S3

---

### Notebook 05: Token/NER Classification

**Domain**: Contract entity extraction for automated validation

**Difficulty**: ⭐⭐⭐ Hard

**Learning Goals**:
- Handle sequence labeling quality issues
- Detect entity boundary errors
- Validate extracted amounts/dates/parties

**Cleanlab Functions Used**:
```python
from cleanlab.token_classification import (
    find_label_issues,
    get_label_quality_scores,
    display_issues
)
```

**YRSN Mapping**:
| Cleanlab Signal | → | YRSN Component |
|-----------------|---|----------------|
| Token label issues | → | N (wrong entity) |
| Boundary errors | → | S (ambiguous span) |
| Low token confidence | → | N or S depending |
| Clean extraction | → | R (trust this value) |

**Collapse Type**: POISONING (wrong amount/date extracted)

**Sample Data Structure**:
```python
{
    "contract_text": "Agreement dated January 15, 2024 for $50,000...",
    "entities": [
        {"text": "January 15, 2024", "label": "DATE", "start": 16, "end": 32},
        {"text": "$50,000", "label": "AMOUNT", "start": 37, "end": 44}
    ],
    "sentence_tokens": ["Agreement", "dated", "January", "15", ",", "2024", ...]
}
```

**AWS Services**: Comprehend (NER), Textract, S3

---

### Notebook 06: Production Pipeline

**Domain**: Full IARS deployment with all components

**Difficulty**: ⭐⭐⭐ Hard

**Learning Goals**:
- Integrate all Cleanlab functions via Datalab
- Implement temperature-aware routing
- Handle multiple collapse types simultaneously
- Production monitoring and metrics

**Cleanlab Functions Used**:
```python
from cleanlab import Datalab
from cleanlab.dataset import (
    overall_label_health_score,
    rank_classes_by_label_quality,
    get_confident_thresholds
)
from cleanlab.data_valuation import data_shapley_knn
```

**YRSN Mapping**:
| Cleanlab Signal | → | YRSN Component |
|-----------------|---|----------------|
| All Datalab issues | → | Comprehensive R/S/N |
| Dataset health | → | Global quality metric |
| Data Shapley | → | Example importance |
| Class thresholds | → | Category-level routing |

**Collapse Types**: ALL (different types for different requests)

**Architecture**:
```
Request → Classifier → Cleanlab Signals → YRSN → Temperature → Router → Stream
            ↓              ↓                ↓
         Bedrock      Label Quality      τ = 1/α
                      OOD Detection
                      Duplicates
```

**AWS Services**: Full stack (S3, Bedrock, SageMaker, Lambda, Step Functions, DynamoDB, CloudWatch)

---

## Progression Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LEARNING PROGRESSION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EASY ─────────────────> MEDIUM ─────────────────> HARD                    │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │   01    │───>│   02    │───>│   03    │───>│   04    │───>│   05    │  │
│  │ Basic   │    │  Text   │    │ Multi-  │    │  RAG    │    │ Token   │  │
│  │ Labels  │    │ Quality │    │Annotator│    │ Context │    │  NER    │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┴──────────────┘        │
│                                   │                                        │
│                                   ▼                                        │
│                            ┌─────────────┐                                 │
│                            │     06      │                                 │
│                            │ Production  │                                 │
│                            │  Pipeline   │                                 │
│                            └─────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Collapse Type Reference

| Collapse Type | Primary Signal | Routing Impact | Recovery Strategy |
|---------------|----------------|----------------|-------------------|
| **POISONING** | High N (noise) | Force RED stream | Expert review, data audit |
| **DISTRACTION** | High S (superfluous) | Yellow + summarization | Context compression |
| **CONFUSION** | High N + S | Force RED stream | Clarification request |
| **CLASH** | Variable S | Yellow + escalation | Multi-approver consensus |

---

## Dataset Requirements

### Minimum Dataset Sizes

| Notebook | Min Examples | Min Classes | Notes |
|----------|-------------|-------------|-------|
| 01 | 500 | 3 | Need enough for label quality estimation |
| 02 | 1000 | 5+ | Text needs variety |
| 03 | 200 | 3 | Multiple annotations per example |
| 04 | 500 | 10+ | Diverse policy documents |
| 05 | 300 | 5 | Token-level labels |
| 06 | 2000+ | 10+ | Production-scale |

### Suggested Open Datasets

1. **Notebook 01-02**: Custom synthetic approval data (provided in repo)
2. **Notebook 03**: Movie review multi-annotator dataset
3. **Notebook 04**: MS MARCO or custom policy corpus
4. **Notebook 05**: CoNLL-2003 or custom contract NER
5. **Notebook 06**: Combination of above

---

## Temperature Settings by Notebook

| Notebook | Default Mode | τ Range | Rationale |
|----------|-------------|---------|-----------|
| 01 | FIXED_MID | 1.0 | Learning baseline |
| 02 | ADAPTIVE | 0.5-2.0 | Text quality varies |
| 03 | ADAPTIVE | 0.3-3.0 | High disagreement = high τ |
| 04 | FIXED_LOOSE | 2.0 | RAG uncertainty |
| 05 | FIXED_TIGHT | 0.3 | Entity extraction critical |
| 06 | ANNEALING | 2.0→0.5 | Production warm-up |

---

## Success Metrics by Notebook

| Notebook | Primary Metric | Target | Secondary |
|----------|---------------|--------|-----------|
| 01 | Accuracy on held-out labels | 95% | AUC-ROC |
| 02 | F1 on clean vs noisy text | 90% | Precision |
| 03 | Agreement with expert consensus | 92% | Kappa |
| 04 | Retrieval relevance | MRR@10 > 0.8 | nDCG |
| 05 | Entity F1 | 88% | Exact match |
| 06 | Automation rate at 99% precision | 60%+ | Latency P99 |

---

*Document Version: 1.0*
*Last Updated: 2024*
*YRSN-IARS Integration Team*
