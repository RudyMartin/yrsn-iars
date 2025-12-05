# YRSN-IARS Master Specification

## Executive Summary

This specification defines how to integrate **Cleanlab** (data quality detection) with **YRSN** (context decomposition) to create a unified framework for **LLM context quality assessment**.

### The Problem

LLMs suffer from context quality issues:
1. **Noisy training data** - Mislabeled examples degrade fine-tuning
2. **Poor retrieval context** - RAG returns irrelevant documents
3. **Context collapse** - Model outputs degrade under certain conditions
4. **Prompt ambiguity** - Unclear instructions cause inconsistent behavior

### The Solution

Combine two complementary approaches:

| Framework | Strength | Gap |
|-----------|----------|-----|
| **Cleanlab** | Detects label errors, outliers, duplicates | Binary (correct/incorrect), no ternary decomposition |
| **YRSN** | Decomposes into R/S/N, detects collapse types | Needs quality signals from data |

**Together**: Cleanlab provides the signals, YRSN provides the framework.

---

## Part 1: Cleanlab Signal Inventory

### 1.1 Per-Example Scores (Primary Signals)

| Signal | Type | Range | Semantics |
|--------|------|-------|-----------|
| `label_quality_score` | float | [0,1] | 1=clean label, 0=likely wrong |
| `self_confidence` | float | [0,1] | P(model agrees with label) |
| `normalized_margin` | float | [0,1] | Gap between top-2 predictions |
| `confidence_weighted_entropy` | float | [0,1] | Uncertainty-adjusted confidence |
| `ood_score` | float | [0,1] | 1=in-distribution, 0=outlier |
| `data_shapley_score` | float | [0,1] | Training value of example |
| `is_label_issue` | bool | T/F | Flagged as mislabeled |
| `is_near_duplicate` | bool | T/F | Similar to another example |
| `near_duplicate_cluster_id` | int | ≥0 | Cluster of duplicates |

### 1.2 Dataset-Level Metrics

| Metric | Type | Semantics |
|--------|------|-----------|
| `overall_label_health_score` | float [0,1] | Dataset-wide quality |
| `num_label_issues` | int | Count of problematic examples |
| `confident_joint` | int[K,K] | Label confusion matrix |
| `noise_matrix` | float[K,K] | P(given\|true) per class |
| `inv_noise_matrix` | float[K,K] | P(true\|given) per class |

### 1.3 Text-Specific Signals

| Signal | Type | Range | Semantics |
|--------|------|-------|-----------|
| `toxic_score` | float | [0,1] | Hateful/aggressive content |
| `pii_score` | float | [0,1] | Contains personal info |
| `non_english_score` | float | [0,1] | Foreign language/garbage |
| `informal_score` | float | [0,1] | Grammar/spelling issues |

### 1.4 Multi-Annotator Signals

| Signal | Type | Semantics |
|--------|------|-----------|
| `consensus_label` | int | Best estimate of true label |
| `consensus_quality_score` | float [0,1] | Confidence in consensus |
| `annotator_agreement` | float [0,1] | Inter-annotator agreement |
| `annotator_weights` | float[M] | Per-annotator reliability |

---

## Part 2: YRSN Mapping Rules

### 2.1 The Y=R+S+N Framework

```
Y (Yield) = R (Relevant) + S (Superfluous) + N (Noise)

Where:
- R = Useful signal that helps the task
- S = Harmless but unnecessary content
- N = Harmful interference that hurts performance
```

### 2.2 Mapping Cleanlab → YRSN

#### Per-Example Mapping

```python
def cleanlab_to_yrsn(
    label_quality: float,
    normalized_margin: float,
    normalized_entropy: float,
    ood_score: float,
    is_duplicate: bool,
    data_shapley: float
) -> tuple[float, float, float]:
    """
    Convert Cleanlab scores to YRSN R, S, N components.

    Logic:
    - N (Noise): Mislabeled OR out-of-distribution OR harmful
    - S (Superfluous): Duplicate OR ambiguous OR low-value but correct
    - R (Relevant): Everything else (clean, useful, in-distribution)
    """

    # N: Sources of harmful interference
    n_mislabel = 1 - label_quality           # Wrong label → noise
    n_ood = 1 - ood_score                    # Out-of-distribution → noise
    n_harmful = max(0, 0.5 - data_shapley)   # Negative value → noise

    N = min(1.0, n_mislabel + 0.5 * n_ood + n_harmful)

    # S: Sources of harmless redundancy
    s_duplicate = 1.0 if is_duplicate else 0.0
    s_ambiguous = normalized_entropy * (1 - N)  # High entropy but not noise
    s_margin = (1 - normalized_margin) * 0.5    # Low margin = uncertain

    S = min(1.0 - N, s_duplicate * 0.8 + s_ambiguous * 0.5 + s_margin * 0.3)

    # R: What remains
    R = 1.0 - N - S

    return R, S, N
```

#### Dataset-Level Mapping

```python
def confident_joint_to_yrsn(confident_joint: np.ndarray) -> tuple[float, float, float]:
    """
    Convert confident joint matrix to dataset-level YRSN.

    - Diagonal = correctly labeled = R
    - Off-diagonal = mislabeled = N
    - S requires additional analysis (duplicates, ambiguous)
    """
    total = confident_joint.sum()
    correct = np.trace(confident_joint)
    incorrect = total - correct

    R = correct / total      # Fraction correctly labeled
    N = incorrect / total    # Fraction mislabeled
    S = 0.0                  # Requires duplicate/ambiguity analysis

    return R, S, N
```

### 2.3 Collapse Type Detection

| Cleanlab Pattern | YRSN Collapse | Detection Rule |
|------------------|---------------|----------------|
| Low `label_quality` across dataset | **POISONING** | `mean(label_quality) < 0.5` |
| High `normalized_entropy` | **DISTRACTION** | `mean(entropy) > 0.6` |
| Low quality AND high entropy | **CONFUSION** | Both conditions |
| High variance in `annotator_agreement` | **CLASH** | `std(agreement) > 0.3` |
| Many duplicates | **DISTRACTION** | `duplicate_rate > 0.2` |

### 2.4 Text Quality → YRSN

```python
def text_quality_to_yrsn(
    toxic_score: float,
    pii_score: float,
    non_english_score: float,
    informal_score: float
) -> tuple[float, float, float]:
    """
    Convert text quality scores to YRSN.

    - Toxic/PII/non-English → Noise (harmful)
    - Informal → Superfluous (not ideal but not harmful)
    """
    N = max(toxic_score, pii_score, non_english_score)
    S = informal_score * (1 - N)
    R = 1.0 - N - S

    return R, S, N
```

---

## Part 3: Datasets for IARS Notebooks

### 3.1 Text Classification Datasets

| Dataset | Size | Classes | Source | YRSN Use Case |
|---------|------|---------|--------|---------------|
| **BANKING77** | 13K | 77 intents | HuggingFace | Intent classification, customer service |
| **AG News** | 120K | 4 topics | torchtext | News categorization |
| **IMDB** | 50K | 2 (pos/neg) | HuggingFace | Sentiment analysis |
| **20 Newsgroups** | 18K | 20 topics | sklearn | Document classification |
| **Amazon Reviews** | 3.6M | 5 stars | AWS | Product review quality |

### 3.2 Instruction/Prompt Datasets

| Dataset | Size | Source | YRSN Use Case |
|---------|------|--------|---------------|
| **Alpaca** | 52K | Stanford | Instruction quality |
| **Dolly** | 15K | Databricks | Human-written instructions |
| **OpenAssistant** | 66K | LAION | Conversation quality |
| **ShareGPT** | 90K | Community | Multi-turn quality |

### 3.3 RAG/Document Datasets

| Dataset | Size | Source | YRSN Use Case |
|---------|------|--------|---------------|
| **MS MARCO** | 8.8M | Microsoft | Passage retrieval quality |
| **Natural Questions** | 307K | Google | QA context quality |
| **HotpotQA** | 113K | CMU | Multi-hop reasoning |
| **SEC EDGAR** | Variable | SEC | Financial document quality |

### 3.4 Multi-Annotator Datasets

| Dataset | Annotators | Source | YRSN Use Case |
|---------|------------|--------|---------------|
| **CrowdFlower** | Variable | Kaggle | Annotator disagreement |
| **SNLI** | 5 | Stanford | NLI label quality |

### 3.5 AWS-Native Datasets

| Dataset | Service | Use Case |
|---------|---------|----------|
| **Amazon Reviews** | S3 Public | Product classification |
| **Common Crawl** | S3 Public | Web text quality |
| **CORD-19** | S3 Public | Scientific document quality |

---

## Part 4: AWS Infrastructure

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │   S3     │───▶│  SageMaker   │───▶│   Amazon Bedrock    │   │
│  │ Datasets │    │  Processing  │    │  (Claude/Titan)     │   │
│  └──────────┘    └──────────────┘    └─────────────────────┘   │
│       │                │                       │                │
│       │                ▼                       ▼                │
│       │         ┌──────────────┐    ┌─────────────────────┐   │
│       │         │   Cleanlab   │    │   YRSN Analysis     │   │
│       │         │   Analysis   │    │   (Collapse Det.)   │   │
│       │         └──────────────┘    └─────────────────────┘   │
│       │                │                       │                │
│       │                └───────────┬───────────┘                │
│       │                            ▼                            │
│       │                   ┌──────────────┐                     │
│       └──────────────────▶│   Results    │                     │
│                           │   (S3/DDB)   │                     │
│                           └──────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Service Configuration

#### Amazon Bedrock (LLM Inference)

```python
# bedrock_config.py
BEDROCK_CONFIG = {
    "region": "us-east-1",
    "models": {
        "claude_sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude_haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "titan_embed": "amazon.titan-embed-text-v1",
        "titan_express": "amazon.titan-text-express-v1"
    },
    "inference_params": {
        "max_tokens": 4096,
        "temperature": 0.0,  # Deterministic for quality analysis
        "top_p": 1.0
    }
}
```

#### Amazon SageMaker (Model Training)

```python
# sagemaker_config.py
SAGEMAKER_CONFIG = {
    "processing_instance": "ml.m5.xlarge",
    "training_instance": "ml.g4dn.xlarge",  # GPU for embeddings
    "endpoint_instance": "ml.m5.large",
    "framework": "pytorch",
    "framework_version": "2.0.0",
    "python_version": "py310"
}
```

#### Amazon S3 (Data Storage)

```python
# s3_config.py
S3_CONFIG = {
    "bucket": "yrsn-iars-data",
    "prefixes": {
        "raw": "datasets/raw/",
        "processed": "datasets/processed/",
        "cleanlab": "analysis/cleanlab/",
        "yrsn": "analysis/yrsn/",
        "models": "models/",
        "results": "results/"
    }
}
```

### 4.3 IAM Permissions Required

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "arn:aws:bedrock:*:*:model/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateProcessingJob",
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateEndpoint",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::yrsn-iars-data",
                "arn:aws:s3:::yrsn-iars-data/*"
            ]
        }
    ]
}
```

---

## Part 5: Notebook Series Specification

### Notebook 1: Cleanlab Basics for LLM Data

**File**: `01_cleanlab_basics.ipynb`

**Objective**: Understand Cleanlab signals on text data

**Sections**:
1. Install & Import
2. Load BANKING77 dataset
3. Train text classifier (sentence-transformers + LogisticRegression)
4. Get predicted probabilities via cross-validation
5. Compute label quality scores
6. Identify label issues
7. Visualize problematic examples
8. Export cleaned dataset

**Key Code**:
```python
from sklearn.model_selection import cross_val_predict
from sentence_transformers import SentenceTransformer
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

# Embed text
encoder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = encoder.encode(texts)

# Train classifier and get OOF predictions
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
pred_probs = cross_val_predict(clf, embeddings, labels, cv=5, method='predict_proba')

# Cleanlab analysis
quality_scores = get_label_quality_scores(labels, pred_probs)
issues = find_label_issues(labels, pred_probs)

print(f"Found {issues.sum()} label issues out of {len(labels)} examples")
```

---

### Notebook 2: YRSN Decomposition Fundamentals

**File**: `02_yrsn_decomposition.ipynb`

**Objective**: Understand Y=R+S+N framework and collapse detection

**Sections**:
1. YRSN Theory Overview
2. Manual R/S/N calculation examples
3. Collapse type definitions
4. Using detect_collapse()
5. Paradigm remedies
6. Visualizing YRSN distributions

**Key Code**:
```python
from yrsn_context import detect_collapse, CollapseType

# Example: High noise scenario
analysis = detect_collapse(R=0.3, S=0.2, N=0.5)
print(f"Collapse: {analysis.collapse_type}")      # POISONING
print(f"Severity: {analysis.severity}")           # HIGH
print(f"Remedy: {analysis.paradigm_remedy}")      # Iterative Refinement

# Example: High superfluous scenario
analysis = detect_collapse(R=0.4, S=0.5, N=0.1)
print(f"Collapse: {analysis.collapse_type}")      # DISTRACTION
print(f"Remedy: {analysis.paradigm_remedy}")      # Bit-Slicing
```

---

### Notebook 3: Cleanlab → YRSN Integration

**File**: `03_integration_adapter.ipynb`

**Objective**: Build and test the CleanlabAdapter

**Sections**:
1. Design the adapter interface
2. Implement per-example mapping
3. Implement dataset-level mapping
4. Test on BANKING77
5. Validate YRSN distributions
6. Compare to manual YRSN labels

**Key Code**:
```python
from yrsn_iars.adapters import CleanlabAdapter

adapter = CleanlabAdapter()

# Per-example conversion
R, S, N = adapter.example_to_yrsn(
    label_quality=0.85,
    normalized_margin=0.72,
    normalized_entropy=0.31,
    ood_score=0.95,
    is_duplicate=False,
    data_shapley=0.68
)
print(f"R={R:.3f}, S={S:.3f}, N={N:.3f}")

# Batch conversion
yrsn_df = adapter.batch_to_yrsn(cleanlab_results_df)

# Dataset-level
R_total, S_total, N_total = adapter.dataset_to_yrsn(confident_joint)
```

---

### Notebook 4: LLM Context Quality Pipeline

**File**: `04_llm_context_quality.ipynb`

**Objective**: End-to-end pipeline for LLM training data curation

**Sections**:
1. Load instruction dataset (Alpaca)
2. Generate embeddings via Bedrock Titan
3. Train quality classifier
4. Apply Cleanlab analysis
5. Convert to YRSN
6. Filter by collapse risk
7. Export curated dataset
8. Compare fine-tuning results

**Key Code**:
```python
from yrsn_iars.pipelines import LLMContextQualityPipeline
from yrsn_iars.aws import BedrockEmbedder

# Initialize pipeline
pipeline = LLMContextQualityPipeline(
    embedder=BedrockEmbedder(model_id="amazon.titan-embed-text-v1"),
    cleanlab_config={"filter_by": "low_self_confidence"},
    yrsn_config={"collapse_threshold": 0.6}
)

# Run analysis
results = pipeline.analyze(
    texts=instructions,
    labels=categories,
    return_yrsn=True
)

# Filter by quality
clean_data = pipeline.filter_by_quality(
    data=dataset,
    min_R=0.5,      # Minimum relevance
    max_N=0.3,      # Maximum noise
    max_S=0.4       # Maximum superfluous
)

print(f"Kept {len(clean_data)} / {len(dataset)} examples")
```

---

### Notebook 5: RAG Context Curation

**File**: `05_rag_curation.ipynb`

**Objective**: Apply YRSN to RAG retrieval quality

**Sections**:
1. Load document corpus (MS MARCO subset)
2. Build vector index
3. Retrieve for sample queries
4. Score retrieved contexts with Cleanlab
5. Decompose into R/S/N
6. Detect collapse per query
7. Re-rank by YRSN quality
8. Compare retrieval metrics

**Key Code**:
```python
from yrsn_iars.pipelines import RAGQualityPipeline

pipeline = RAGQualityPipeline(
    retriever=your_retriever,
    reranker=None,  # Optional
    yrsn_config={"paradigm_on_collapse": True}
)

# Analyze retrieved context
for query in queries:
    retrieved_docs = retriever.retrieve(query, k=10)

    analysis = pipeline.analyze_retrieval(
        query=query,
        documents=retrieved_docs
    )

    print(f"Query: {query[:50]}...")
    print(f"  R={analysis.R:.2f}, S={analysis.S:.2f}, N={analysis.N:.2f}")
    print(f"  Collapse: {analysis.collapse_type}")

    if analysis.collapse_type != CollapseType.NONE:
        # Apply paradigm remedy
        fixed_docs = pipeline.apply_remedy(
            documents=retrieved_docs,
            paradigm=analysis.paradigm_remedy
        )
```

---

### Notebook 6: AWS Production Pipeline

**File**: `06_aws_production.ipynb`

**Objective**: Deploy YRSN-IARS on AWS infrastructure

**Sections**:
1. Set up S3 bucket and IAM
2. Create SageMaker processing job for Cleanlab
3. Create YRSN analysis Lambda
4. Build Step Functions workflow
5. Deploy real-time quality endpoint
6. Monitor with CloudWatch
7. Cost optimization

**Key Code**:
```python
import boto3
from yrsn_iars.aws import (
    create_processing_job,
    create_yrsn_lambda,
    create_step_function
)

# 1. SageMaker Processing Job
processing_job = create_processing_job(
    job_name="yrsn-iars-cleanlab",
    input_s3="s3://yrsn-iars-data/datasets/raw/",
    output_s3="s3://yrsn-iars-data/analysis/cleanlab/",
    script="cleanlab_processor.py",
    instance_type="ml.m5.xlarge"
)

# 2. Lambda for YRSN Analysis
lambda_arn = create_yrsn_lambda(
    function_name="yrsn-collapse-detector",
    handler="lambda_handler.detect_collapse",
    memory_mb=512,
    timeout_sec=30
)

# 3. Step Functions Workflow
workflow_arn = create_step_function(
    name="yrsn-iars-pipeline",
    definition={
        "StartAt": "CleanlabAnalysis",
        "States": {
            "CleanlabAnalysis": {
                "Type": "Task",
                "Resource": f"arn:aws:states:::sagemaker:createProcessingJob.sync",
                "Next": "YRSNDecomposition"
            },
            "YRSNDecomposition": {
                "Type": "Task",
                "Resource": lambda_arn,
                "End": True
            }
        }
    }
)
```

---

## Part 6: Code Module Specifications

### 6.1 CleanlabAdapter

**Location**: `src/yrsn_iars/adapters/cleanlab_adapter.py`

```python
"""
CleanlabAdapter: Convert Cleanlab outputs to YRSN R/S/N decomposition.

This adapter bridges Cleanlab's data quality signals with YRSN's
context decomposition framework for LLM quality assessment.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class YRSNResult:
    """Result of YRSN decomposition."""
    R: float  # Relevant [0, 1]
    S: float  # Superfluous [0, 1]
    N: float  # Noise [0, 1]

    def __post_init__(self):
        total = self.R + self.S + self.N
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(f"R + S + N must equal 1.0, got {total}")

    @property
    def quality_score(self) -> float:
        """Y-score: R + 0.5*S (from YRSN theory)."""
        return self.R + 0.5 * self.S

    @property
    def risk_score(self) -> float:
        """Collapse risk: S + 1.5*N."""
        return self.S + 1.5 * self.N


class CleanlabAdapter:
    """
    Adapter to convert Cleanlab data quality outputs to YRSN decomposition.

    Cleanlab provides signals about data quality (label errors, outliers, etc.)
    YRSN decomposes context into Relevant + Superfluous + Noise.

    This adapter maps:
    - Label issues → N (Noise)
    - Duplicates/ambiguity → S (Superfluous)
    - Clean, useful data → R (Relevant)

    Parameters
    ----------
    noise_weight : float, default=1.0
        Weight for noise component from label quality
    ood_weight : float, default=0.5
        Weight for out-of-distribution contribution to noise
    duplicate_weight : float, default=0.8
        Weight for duplicate contribution to superfluous
    entropy_weight : float, default=0.5
        Weight for entropy contribution to superfluous
    """

    def __init__(
        self,
        noise_weight: float = 1.0,
        ood_weight: float = 0.5,
        duplicate_weight: float = 0.8,
        entropy_weight: float = 0.5
    ):
        self.noise_weight = noise_weight
        self.ood_weight = ood_weight
        self.duplicate_weight = duplicate_weight
        self.entropy_weight = entropy_weight

    def example_to_yrsn(
        self,
        label_quality: float,
        normalized_margin: Optional[float] = None,
        normalized_entropy: Optional[float] = None,
        ood_score: Optional[float] = None,
        is_duplicate: bool = False,
        data_shapley: Optional[float] = None,
        **kwargs
    ) -> YRSNResult:
        """
        Convert a single example's Cleanlab scores to YRSN.

        Parameters
        ----------
        label_quality : float
            Cleanlab label quality score [0, 1]
        normalized_margin : float, optional
            Margin between top predictions [0, 1]
        normalized_entropy : float, optional
            Prediction entropy [0, 1]
        ood_score : float, optional
            Out-of-distribution score [0, 1] (1 = in-distribution)
        is_duplicate : bool
            Whether example is near-duplicate
        data_shapley : float, optional
            Data valuation score [0, 1]

        Returns
        -------
        YRSNResult
            Decomposition with R, S, N components
        """
        # === NOISE CALCULATION ===
        # Primary: inverse of label quality
        n_label = (1 - label_quality) * self.noise_weight

        # Secondary: out-of-distribution examples
        n_ood = 0.0
        if ood_score is not None:
            n_ood = (1 - ood_score) * self.ood_weight

        # Tertiary: harmful examples (negative Shapley value)
        n_harmful = 0.0
        if data_shapley is not None and data_shapley < 0.5:
            n_harmful = (0.5 - data_shapley) * 0.5

        N = min(1.0, n_label + n_ood + n_harmful)

        # === SUPERFLUOUS CALCULATION ===
        remaining = 1.0 - N

        # Duplicates are redundant
        s_duplicate = self.duplicate_weight if is_duplicate else 0.0

        # High entropy (but not noise) indicates ambiguity
        s_entropy = 0.0
        if normalized_entropy is not None:
            s_entropy = normalized_entropy * self.entropy_weight * (1 - N)

        # Low margin indicates uncertainty
        s_margin = 0.0
        if normalized_margin is not None:
            s_margin = (1 - normalized_margin) * 0.3 * (1 - N)

        S = min(remaining, s_duplicate + s_entropy + s_margin)

        # === RELEVANT CALCULATION ===
        R = 1.0 - N - S

        return YRSNResult(R=R, S=S, N=N)

    def batch_to_yrsn(
        self,
        cleanlab_df: pd.DataFrame,
        label_quality_col: str = "label_quality",
        margin_col: Optional[str] = "normalized_margin",
        entropy_col: Optional[str] = "normalized_entropy",
        ood_col: Optional[str] = "ood_score",
        duplicate_col: Optional[str] = "is_near_duplicate",
        shapley_col: Optional[str] = "data_shapley"
    ) -> pd.DataFrame:
        """
        Convert batch of Cleanlab results to YRSN.

        Parameters
        ----------
        cleanlab_df : pd.DataFrame
            DataFrame with Cleanlab output columns
        *_col : str
            Column name mappings

        Returns
        -------
        pd.DataFrame
            Original data with R, S, N, quality_score, risk_score columns
        """
        results = []

        for idx, row in cleanlab_df.iterrows():
            yrsn = self.example_to_yrsn(
                label_quality=row.get(label_quality_col, 0.5),
                normalized_margin=row.get(margin_col) if margin_col else None,
                normalized_entropy=row.get(entropy_col) if entropy_col else None,
                ood_score=row.get(ood_col) if ood_col else None,
                is_duplicate=row.get(duplicate_col, False) if duplicate_col else False,
                data_shapley=row.get(shapley_col) if shapley_col else None
            )
            results.append({
                "R": yrsn.R,
                "S": yrsn.S,
                "N": yrsn.N,
                "quality_score": yrsn.quality_score,
                "risk_score": yrsn.risk_score
            })

        yrsn_df = pd.DataFrame(results)
        return pd.concat([cleanlab_df.reset_index(drop=True), yrsn_df], axis=1)

    def dataset_to_yrsn(
        self,
        confident_joint: np.ndarray,
        duplicate_rate: float = 0.0
    ) -> YRSNResult:
        """
        Convert dataset-level confident joint to YRSN.

        Parameters
        ----------
        confident_joint : np.ndarray
            K x K confident joint matrix from Cleanlab
        duplicate_rate : float
            Fraction of examples that are duplicates [0, 1]

        Returns
        -------
        YRSNResult
            Dataset-level YRSN decomposition
        """
        total = confident_joint.sum()
        correct = np.trace(confident_joint)
        incorrect = total - correct

        # N: Fraction with label issues
        N = incorrect / total

        # S: Duplicates + some portion of correct but ambiguous
        # (Ambiguity estimated from off-diagonal confusion)
        off_diag = confident_joint - np.diag(np.diag(confident_joint))
        confusion_rate = off_diag.sum() / total / 2  # Symmetric confusion

        S = duplicate_rate * self.duplicate_weight + confusion_rate * 0.3
        S = min(S, 1.0 - N)

        # R: What remains
        R = 1.0 - N - S

        return YRSNResult(R=R, S=S, N=N)

    def text_quality_to_yrsn(
        self,
        toxic_score: float = 0.0,
        pii_score: float = 0.0,
        non_english_score: float = 0.0,
        informal_score: float = 0.0
    ) -> YRSNResult:
        """
        Convert text quality scores to YRSN.

        Text-specific issues from Cleanlab Studio:
        - Toxic, PII, non-English → Noise (harmful)
        - Informal → Superfluous (not ideal)

        Parameters
        ----------
        toxic_score : float
            Toxicity score [0, 1]
        pii_score : float
            PII detection score [0, 1]
        non_english_score : float
            Non-English/garbage score [0, 1]
        informal_score : float
            Informal language score [0, 1]

        Returns
        -------
        YRSNResult
            Text quality as YRSN decomposition
        """
        # N: Harmful content
        N = max(toxic_score, pii_score, non_english_score)

        # S: Not ideal but not harmful
        S = informal_score * (1 - N)

        # R: Clean content
        R = 1.0 - N - S

        return YRSNResult(R=R, S=S, N=N)
```

### 6.2 LLMContextQualityPipeline

**Location**: `src/yrsn_iars/pipelines/llm_context_quality.py`

```python
"""
LLMContextQualityPipeline: End-to-end pipeline for LLM data quality.

Combines Cleanlab analysis with YRSN decomposition for curating
high-quality LLM training/evaluation data.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol
import numpy as np
import pandas as pd

from ..adapters.cleanlab_adapter import CleanlabAdapter, YRSNResult


class Embedder(Protocol):
    """Protocol for text embedding models."""
    def encode(self, texts: List[str]) -> np.ndarray: ...


@dataclass
class PipelineConfig:
    """Configuration for LLM context quality pipeline."""

    # Cleanlab settings
    filter_by: str = "low_self_confidence"
    frac_noise: float = 1.0
    n_jobs: int = -1

    # YRSN thresholds
    min_R: float = 0.5
    max_N: float = 0.3
    max_S: float = 0.4
    collapse_threshold: float = 0.6

    # Pipeline behavior
    return_scores: bool = True
    detect_collapse: bool = True


@dataclass
class QualityAnalysis:
    """Result of quality analysis."""

    # Per-example scores
    label_quality: np.ndarray
    R: np.ndarray
    S: np.ndarray
    N: np.ndarray

    # Flags
    is_label_issue: np.ndarray
    is_high_quality: np.ndarray

    # Dataset-level
    dataset_R: float
    dataset_S: float
    dataset_N: float
    collapse_type: str
    collapse_severity: str
    paradigm_remedy: str

    # Counts
    n_total: int
    n_issues: int
    n_high_quality: int


class LLMContextQualityPipeline:
    """
    End-to-end pipeline for LLM data quality assessment.

    Steps:
    1. Embed text using provided embedder
    2. Train classifier and get cross-validated predictions
    3. Run Cleanlab analysis
    4. Convert to YRSN decomposition
    5. Detect collapse type
    6. Filter by quality thresholds

    Parameters
    ----------
    embedder : Embedder
        Text embedding model (e.g., SentenceTransformer, Bedrock Titan)
    classifier : sklearn classifier, optional
        Classifier for label quality (default: LogisticRegression)
    config : PipelineConfig, optional
        Pipeline configuration
    """

    def __init__(
        self,
        embedder: Embedder,
        classifier: Any = None,
        config: Optional[PipelineConfig] = None
    ):
        self.embedder = embedder
        self.classifier = classifier
        self.config = config or PipelineConfig()
        self.adapter = CleanlabAdapter()

        if classifier is None:
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(max_iter=1000, n_jobs=-1)

    def analyze(
        self,
        texts: List[str],
        labels: np.ndarray,
        return_yrsn: bool = True
    ) -> QualityAnalysis:
        """
        Run full quality analysis on text dataset.

        Parameters
        ----------
        texts : List[str]
            Input texts
        labels : np.ndarray
            Labels (integers 0 to K-1)
        return_yrsn : bool
            Whether to compute YRSN decomposition

        Returns
        -------
        QualityAnalysis
            Comprehensive quality analysis results
        """
        from sklearn.model_selection import cross_val_predict
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
        from cleanlab.count import compute_confident_joint

        # Step 1: Embed
        print("Embedding texts...")
        embeddings = self.embedder.encode(texts)

        # Step 2: Cross-validated predictions
        print("Getting cross-validated predictions...")
        pred_probs = cross_val_predict(
            self.classifier,
            embeddings,
            labels,
            cv=5,
            method='predict_proba',
            n_jobs=self.config.n_jobs
        )

        # Step 3: Cleanlab analysis
        print("Running Cleanlab analysis...")
        label_quality = get_label_quality_scores(labels, pred_probs)
        is_label_issue = find_label_issues(
            labels, pred_probs,
            filter_by=self.config.filter_by,
            frac_noise=self.config.frac_noise
        )
        confident_joint = compute_confident_joint(labels, pred_probs)

        # Step 4: YRSN decomposition
        R = np.zeros(len(texts))
        S = np.zeros(len(texts))
        N = np.zeros(len(texts))

        if return_yrsn:
            print("Computing YRSN decomposition...")

            # Compute entropy for each example
            entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10), axis=1)
            max_entropy = np.log(pred_probs.shape[1])
            normalized_entropy = entropy / max_entropy

            # Compute margin
            sorted_probs = np.sort(pred_probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            normalized_margin = (margin + 1) / 2

            for i in range(len(texts)):
                yrsn = self.adapter.example_to_yrsn(
                    label_quality=label_quality[i],
                    normalized_margin=normalized_margin[i],
                    normalized_entropy=normalized_entropy[i]
                )
                R[i], S[i], N[i] = yrsn.R, yrsn.S, yrsn.N

        # Step 5: Dataset-level YRSN
        dataset_yrsn = self.adapter.dataset_to_yrsn(confident_joint)

        # Step 6: Collapse detection
        collapse_type = "NONE"
        collapse_severity = "none"
        paradigm_remedy = "Any"

        if self.config.detect_collapse:
            from yrsn_context import detect_collapse
            analysis = detect_collapse(
                R=dataset_yrsn.R,
                S=dataset_yrsn.S,
                N=dataset_yrsn.N
            )
            collapse_type = analysis.collapse_type.name
            collapse_severity = analysis.severity.value
            paradigm_remedy = analysis.paradigm_remedy

        # Step 7: Quality filtering
        is_high_quality = (
            (R >= self.config.min_R) &
            (N <= self.config.max_N) &
            (S <= self.config.max_S)
        )

        return QualityAnalysis(
            label_quality=label_quality,
            R=R, S=S, N=N,
            is_label_issue=is_label_issue,
            is_high_quality=is_high_quality,
            dataset_R=dataset_yrsn.R,
            dataset_S=dataset_yrsn.S,
            dataset_N=dataset_yrsn.N,
            collapse_type=collapse_type,
            collapse_severity=collapse_severity,
            paradigm_remedy=paradigm_remedy,
            n_total=len(texts),
            n_issues=is_label_issue.sum(),
            n_high_quality=is_high_quality.sum()
        )

    def filter_by_quality(
        self,
        data: pd.DataFrame,
        analysis: QualityAnalysis,
        text_col: str = "text",
        label_col: str = "label"
    ) -> pd.DataFrame:
        """
        Filter dataset by YRSN quality thresholds.

        Parameters
        ----------
        data : pd.DataFrame
            Original dataset
        analysis : QualityAnalysis
            Results from analyze()
        text_col, label_col : str
            Column names

        Returns
        -------
        pd.DataFrame
            Filtered dataset with quality scores
        """
        result = data.copy()
        result["label_quality"] = analysis.label_quality
        result["R"] = analysis.R
        result["S"] = analysis.S
        result["N"] = analysis.N
        result["is_label_issue"] = analysis.is_label_issue
        result["is_high_quality"] = analysis.is_high_quality

        return result[result["is_high_quality"]]

    def get_summary(self, analysis: QualityAnalysis) -> str:
        """Generate human-readable summary."""
        return f"""
LLM Context Quality Analysis Summary
====================================

Dataset Size: {analysis.n_total} examples
Label Issues: {analysis.n_issues} ({100*analysis.n_issues/analysis.n_total:.1f}%)
High Quality: {analysis.n_high_quality} ({100*analysis.n_high_quality/analysis.n_total:.1f}%)

YRSN Decomposition (Dataset-Level):
  R (Relevant):    {analysis.dataset_R:.3f}
  S (Superfluous): {analysis.dataset_S:.3f}
  N (Noise):       {analysis.dataset_N:.3f}

Collapse Analysis:
  Type:     {analysis.collapse_type}
  Severity: {analysis.collapse_severity}
  Remedy:   {analysis.paradigm_remedy}

Quality Distribution (Per-Example):
  Mean R: {analysis.R.mean():.3f} (std: {analysis.R.std():.3f})
  Mean S: {analysis.S.mean():.3f} (std: {analysis.S.std():.3f})
  Mean N: {analysis.N.mean():.3f} (std: {analysis.N.std():.3f})
"""
```

### 6.3 AWS Integration Module

**Location**: `src/yrsn_iars/aws/bedrock.py`

```python
"""
AWS Bedrock integration for YRSN-IARS.

Provides embeddings and LLM inference via Amazon Bedrock.
"""

import json
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockEmbedder:
    """
    Text embedder using Amazon Bedrock Titan Embeddings.

    Parameters
    ----------
    model_id : str
        Bedrock model ID for embeddings
    region : str
        AWS region
    profile : str, optional
        AWS profile name
    """

    EMBEDDING_MODELS = {
        "titan-v1": "amazon.titan-embed-text-v1",
        "titan-v2": "amazon.titan-embed-text-v2:0",
    }

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region: str = "us-east-1",
        profile: Optional[str] = None
    ):
        if boto3 is None:
            raise ImportError("boto3 required: pip install boto3")

        session_kwargs = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime")
        self.model_id = model_id

    def encode(self, texts: List[str], batch_size: int = 25) -> np.ndarray:
        """
        Encode texts to embeddings.

        Parameters
        ----------
        texts : List[str]
            Input texts
        batch_size : int
            Batch size for API calls

        Returns
        -------
        np.ndarray
            Embeddings array of shape (n_texts, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch)
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts."""
        results = []

        for text in texts:
            body = json.dumps({"inputText": text})

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]
            results.append(np.array(embedding))

        return results


class BedrockLLM:
    """
    LLM inference using Amazon Bedrock.

    Supports Claude and Titan models for text generation.

    Parameters
    ----------
    model_id : str
        Bedrock model ID
    region : str
        AWS region
    """

    MODELS = {
        "claude-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "titan-express": "amazon.titan-text-express-v1",
        "titan-lite": "amazon.titan-text-lite-v1",
    }

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-east-1",
        profile: Optional[str] = None
    ):
        if boto3 is None:
            raise ImportError("boto3 required: pip install boto3")

        session_kwargs = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime")
        self.model_id = model_id
        self._is_claude = "claude" in model_id.lower()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Optional[str] = None
    ) -> str:
        """
        Generate text completion.

        Parameters
        ----------
        prompt : str
            Input prompt
        max_tokens : int
            Maximum tokens to generate
        temperature : float
            Sampling temperature
        system : str, optional
            System prompt (Claude only)

        Returns
        -------
        str
            Generated text
        """
        if self._is_claude:
            return self._generate_claude(prompt, max_tokens, temperature, system)
        else:
            return self._generate_titan(prompt, max_tokens, temperature)

    def _generate_claude(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system: Optional[str]
    ) -> str:
        """Generate with Claude models."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system:
            body["system"] = system

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def _generate_titan(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate with Titan models."""
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 1.0
            }
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())
        return response_body["results"][0]["outputText"]


def classify_with_llm(
    llm: BedrockLLM,
    text: str,
    categories: List[str],
    return_probs: bool = True
) -> Dict[str, Any]:
    """
    Classify text using LLM with confidence scores.

    This enables Cleanlab analysis without training a classifier,
    using LLM predictions instead.

    Parameters
    ----------
    llm : BedrockLLM
        LLM client
    text : str
        Text to classify
    categories : List[str]
        Possible categories
    return_probs : bool
        Whether to return probability distribution

    Returns
    -------
    dict
        {"category": str, "confidence": float, "probs": List[float]}
    """
    categories_str = "\n".join(f"- {c}" for c in categories)

    prompt = f"""Classify the following text into exactly one of these categories:
{categories_str}

Text: {text}

Respond with ONLY a JSON object in this exact format:
{{"category": "<category name>", "confidence": <0.0-1.0>}}"""

    response = llm.generate(prompt, max_tokens=100, temperature=0.0)

    try:
        result = json.loads(response.strip())
        category = result["category"]
        confidence = result["confidence"]

        # Build probability distribution
        probs = [0.0] * len(categories)
        if category in categories:
            idx = categories.index(category)
            probs[idx] = confidence
            # Distribute remaining probability
            remaining = 1.0 - confidence
            for i in range(len(probs)):
                if i != idx:
                    probs[i] = remaining / (len(categories) - 1)

        return {
            "category": category,
            "confidence": confidence,
            "probs": probs
        }
    except (json.JSONDecodeError, KeyError):
        # Fallback: uniform distribution
        return {
            "category": categories[0],
            "confidence": 1.0 / len(categories),
            "probs": [1.0 / len(categories)] * len(categories)
        }
```

---

## Part 7: Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_cleanlab_adapter.py

import pytest
import numpy as np
from yrsn_iars.adapters.cleanlab_adapter import CleanlabAdapter, YRSNResult


class TestYRSNResult:
    def test_valid_result(self):
        result = YRSNResult(R=0.6, S=0.3, N=0.1)
        assert result.R == 0.6
        assert result.S == 0.3
        assert result.N == 0.1

    def test_invalid_sum(self):
        with pytest.raises(ValueError):
            YRSNResult(R=0.5, S=0.5, N=0.5)

    def test_quality_score(self):
        result = YRSNResult(R=0.6, S=0.3, N=0.1)
        assert result.quality_score == pytest.approx(0.75)  # 0.6 + 0.5*0.3

    def test_risk_score(self):
        result = YRSNResult(R=0.6, S=0.3, N=0.1)
        assert result.risk_score == pytest.approx(0.45)  # 0.3 + 1.5*0.1


class TestCleanlabAdapter:
    @pytest.fixture
    def adapter(self):
        return CleanlabAdapter()

    def test_high_quality_example(self, adapter):
        result = adapter.example_to_yrsn(
            label_quality=0.95,
            normalized_margin=0.85,
            normalized_entropy=0.15,
            ood_score=0.98,
            is_duplicate=False
        )
        assert result.R > 0.7
        assert result.N < 0.1

    def test_noisy_example(self, adapter):
        result = adapter.example_to_yrsn(
            label_quality=0.2,
            normalized_margin=0.3,
            normalized_entropy=0.8,
            ood_score=0.4
        )
        assert result.N > 0.5
        assert result.R < 0.3

    def test_duplicate_example(self, adapter):
        result = adapter.example_to_yrsn(
            label_quality=0.9,
            is_duplicate=True
        )
        assert result.S > 0.5  # Duplicates are superfluous

    def test_sum_to_one(self, adapter):
        for _ in range(100):
            result = adapter.example_to_yrsn(
                label_quality=np.random.random(),
                normalized_margin=np.random.random(),
                normalized_entropy=np.random.random(),
                ood_score=np.random.random(),
                is_duplicate=np.random.random() > 0.5
            )
            assert np.isclose(result.R + result.S + result.N, 1.0)

    def test_dataset_yrsn_clean(self, adapter):
        # Mostly correct labels (diagonal dominant)
        cj = np.array([
            [90, 5, 5],
            [3, 92, 5],
            [2, 3, 95]
        ])
        result = adapter.dataset_to_yrsn(cj)
        assert result.R > 0.9
        assert result.N < 0.1

    def test_dataset_yrsn_noisy(self, adapter):
        # Many errors (off-diagonal high)
        cj = np.array([
            [50, 25, 25],
            [20, 40, 40],
            [30, 30, 40]
        ])
        result = adapter.dataset_to_yrsn(cj)
        assert result.N > 0.4
```

### 7.2 Integration Tests

```python
# tests/test_integration.py

import pytest
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from yrsn_iars.adapters.cleanlab_adapter import CleanlabAdapter
from yrsn_iars.pipelines.llm_context_quality import (
    LLMContextQualityPipeline,
    PipelineConfig
)


class MockEmbedder:
    """Mock embedder for testing without real model."""
    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        # Use TF-IDF as mock embeddings
        vectorizer = TfidfVectorizer(max_features=self.dim)
        return vectorizer.fit_transform(texts).toarray()


@pytest.fixture
def small_dataset():
    """Load small subset of 20 newsgroups."""
    data = fetch_20newsgroups(
        subset='train',
        categories=['sci.space', 'rec.sport.baseball'],
        remove=('headers', 'footers', 'quotes')
    )
    # Take small sample
    indices = np.random.choice(len(data.data), 100, replace=False)
    return {
        "texts": [data.data[i] for i in indices],
        "labels": data.target[indices]
    }


class TestFullPipeline:
    def test_analyze_returns_valid_yrsn(self, small_dataset):
        pipeline = LLMContextQualityPipeline(
            embedder=MockEmbedder(),
            config=PipelineConfig(n_jobs=1)
        )

        analysis = pipeline.analyze(
            texts=small_dataset["texts"],
            labels=small_dataset["labels"]
        )

        # Check YRSN validity
        assert np.all(analysis.R >= 0) and np.all(analysis.R <= 1)
        assert np.all(analysis.S >= 0) and np.all(analysis.S <= 1)
        assert np.all(analysis.N >= 0) and np.all(analysis.N <= 1)
        assert np.allclose(analysis.R + analysis.S + analysis.N, 1.0)

        # Check dataset-level
        assert 0 <= analysis.dataset_R <= 1
        assert 0 <= analysis.dataset_S <= 1
        assert 0 <= analysis.dataset_N <= 1

    def test_collapse_detection(self, small_dataset):
        pipeline = LLMContextQualityPipeline(
            embedder=MockEmbedder(),
            config=PipelineConfig(detect_collapse=True)
        )

        analysis = pipeline.analyze(
            texts=small_dataset["texts"],
            labels=small_dataset["labels"]
        )

        assert analysis.collapse_type in [
            "NONE", "POISONING", "DISTRACTION", "CONFUSION", "CLASH"
        ]
        assert analysis.paradigm_remedy is not None
```

---

## Part 8: Deployment Checklist

### 8.1 Local Development

```bash
# Clone and setup
git clone https://github.com/RudyMartin/yrsn-iars.git
cd yrsn-iars
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run notebooks
jupyter lab notebooks/
```

### 8.2 AWS Deployment

```bash
# Configure AWS credentials
aws configure

# Create S3 bucket
aws s3 mb s3://yrsn-iars-data

# Deploy Lambda function
cd aws/lambda
./deploy.sh

# Create SageMaker resources
python scripts/setup_sagemaker.py
```

### 8.3 Production Monitoring

- CloudWatch metrics for API latency
- S3 access logs for data usage
- Bedrock usage metrics
- YRSN quality score trending

---

## Appendix A: Signal Reference Card

```
CLEANLAB SIGNALS → YRSN MAPPING QUICK REFERENCE
================================================

Per-Example:
  label_quality     [0,1] → N = 1 - score
  normalized_margin [0,1] → R correlation
  normalized_entropy[0,1] → S + N indicator
  ood_score         [0,1] → N = 1 - score
  data_shapley      [0,1] → R correlation, N if < 0.5
  is_duplicate      bool  → S = 0.8 if True

Dataset-Level:
  confident_joint diagonal → R count
  confident_joint off-diag → N count
  noise_matrix diagonal    → R rate per class
  overall_health_score     → R + 0.5*S estimate

Text-Specific:
  toxic_score       [0,1] → N
  pii_score         [0,1] → N
  non_english_score [0,1] → N
  informal_score    [0,1] → S

Collapse Detection:
  mean(label_quality) < 0.5         → POISONING
  mean(entropy) > 0.6               → DISTRACTION
  low_quality AND high_entropy      → CONFUSION
  std(annotator_agreement) > 0.3    → CLASH
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: YRSN-IARS Team*
