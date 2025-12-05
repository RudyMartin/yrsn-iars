"""
CleanlabAdapter: Convert Cleanlab outputs to YRSN R/S/N decomposition.

This adapter bridges Cleanlab's data quality signals with YRSN's
context decomposition framework for the IARS approval routing system.

Mapping Logic:
- N (Noise): Mislabeled data, out-of-distribution requests, harmful patterns
- S (Superfluous): Duplicates, ambiguous cases, low-value but correct
- R (Relevant): Clean, useful, correctly labeled data

Cleanlab Functions Supported:
- find_label_issues() → N (label quality)
- get_label_quality_scores() → N (per-example quality)
- get_confident_thresholds() → Per-class reliability
- rank_classes_by_label_quality() → Class-level S/N
- find_overlapping_classes() → Class confusion → N
- overall_label_health_score() → Global R estimate
- OutOfDistribution.fit_score() → Direct N signal
- data_shapley_knn() → R/S distinction
- is_near_duplicate → S (redundancy)
- normalized_entropy → S (ambiguity)
- Datalab unified interface → All signals
- multiannotator consensus → Multi-approver scenarios
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
from enum import Enum
import numpy as np
import pandas as pd

# Type checking imports for optional Cleanlab dependency
if TYPE_CHECKING:
    try:
        from cleanlab import Datalab
    except ImportError:
        Datalab = None


class CollapseType(Enum):
    """Context collapse types from YRSN framework."""
    NONE = "none"
    POISONING = "poisoning"      # High N - misinformation
    DISTRACTION = "distraction"  # High S - information overload
    CONFUSION = "confusion"      # High N + S - contradictory
    CLASH = "clash"              # Variable S - source conflicts


@dataclass
class YRSNResult:
    """Result of YRSN decomposition."""
    R: float  # Relevant [0, 1]
    S: float  # Superfluous [0, 1]
    N: float  # Noise [0, 1]

    def __post_init__(self):
        total = self.R + self.S + self.N
        if not np.isclose(total, 1.0, atol=0.01):
            # Normalize if needed
            self.R = self.R / total
            self.S = self.S / total
            self.N = self.N / total

    @property
    def quality_score(self) -> float:
        """Y-score: R + 0.5*S (from YRSN theory)."""
        return self.R + 0.5 * self.S

    @property
    def risk_score(self) -> float:
        """Collapse risk: S + 1.5*N."""
        return self.S + 1.5 * self.N

    @property
    def collapse_type(self) -> CollapseType:
        """Detect collapse type from R/S/N ratios."""
        if self.N > 0.3 and self.S > 0.25:
            return CollapseType.CONFUSION
        elif self.N > 0.3:
            return CollapseType.POISONING
        elif self.S > 0.4:
            return CollapseType.DISTRACTION
        elif self.risk_score > 0.6:
            return CollapseType.CLASH
        return CollapseType.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "R": self.R,
            "S": self.S,
            "N": self.N,
            "quality_score": self.quality_score,
            "risk_score": self.risk_score,
            "collapse_type": self.collapse_type.value
        }


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
        R = max(0.0, 1.0 - N - S)

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
                normalized_margin=row.get(margin_col) if margin_col and margin_col in row else None,
                normalized_entropy=row.get(entropy_col) if entropy_col and entropy_col in row else None,
                ood_score=row.get(ood_col) if ood_col and ood_col in row else None,
                is_duplicate=row.get(duplicate_col, False) if duplicate_col and duplicate_col in row else False,
                data_shapley=row.get(shapley_col) if shapley_col and shapley_col in row else None
            )
            results.append(yrsn.to_dict())

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
        off_diag = confident_joint - np.diag(np.diag(confident_joint))
        confusion_rate = off_diag.sum() / total / 2

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

    def compute_confidence_score(
        self,
        yrsn: YRSNResult,
        classifier_confidence: float,
        label_quality: Optional[float] = None,
        ood_score: Optional[float] = None,
        w_classifier: float = 0.4,
        w_context: float = 0.3,
        w_label: float = 0.15,
        w_ood: float = 0.15
    ) -> float:
        """
        Compute overall confidence score (Cs) for routing.

        Combines classifier confidence with YRSN context quality.

        Parameters
        ----------
        yrsn : YRSNResult
            YRSN decomposition result
        classifier_confidence : float
            Model's prediction confidence [0, 1]
        label_quality : float, optional
            Training data quality for this category
        ood_score : float, optional
            Out-of-distribution score
        w_* : float
            Component weights (should sum to 1)

        Returns
        -------
        float
            Confidence score Cs in [0, 1]
        """
        # Context contribution
        context_score = yrsn.quality_score

        # Cleanlab contribution
        label_score = label_quality if label_quality is not None else 0.8
        ood = ood_score if ood_score is not None else 0.9

        Cs = (
            w_classifier * classifier_confidence +
            w_context * context_score +
            w_label * label_score +
            w_ood * ood
        )

        return min(1.0, max(0.0, Cs))

    # =========================================================================
    # VECTORIZED BATCH PROCESSING
    # =========================================================================

    def batch_to_yrsn_vectorized(
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
        Vectorized YRSN conversion for large datasets.

        Uses numpy broadcasting instead of iterrows() for 10-100x speedup
        on large DataFrames.

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
        n = len(cleanlab_df)

        # Extract columns with defaults
        label_quality = cleanlab_df[label_quality_col].values if label_quality_col in cleanlab_df else np.full(n, 0.5)

        margin = cleanlab_df[margin_col].values if margin_col and margin_col in cleanlab_df else None
        entropy = cleanlab_df[entropy_col].values if entropy_col and entropy_col in cleanlab_df else None
        ood = cleanlab_df[ood_col].values if ood_col and ood_col in cleanlab_df else None
        is_dup = cleanlab_df[duplicate_col].values if duplicate_col and duplicate_col in cleanlab_df else np.zeros(n, dtype=bool)
        shapley = cleanlab_df[shapley_col].values if shapley_col and shapley_col in cleanlab_df else None

        # === VECTORIZED NOISE CALCULATION ===
        n_label = (1 - label_quality) * self.noise_weight

        n_ood = np.zeros(n)
        if ood is not None:
            n_ood = (1 - ood) * self.ood_weight

        n_harmful = np.zeros(n)
        if shapley is not None:
            mask = shapley < 0.5
            n_harmful[mask] = (0.5 - shapley[mask]) * 0.5

        N = np.minimum(1.0, n_label + n_ood + n_harmful)

        # === VECTORIZED SUPERFLUOUS CALCULATION ===
        remaining = 1.0 - N

        s_duplicate = np.where(is_dup, self.duplicate_weight, 0.0)

        s_entropy = np.zeros(n)
        if entropy is not None:
            s_entropy = entropy * self.entropy_weight * (1 - N)

        s_margin = np.zeros(n)
        if margin is not None:
            s_margin = (1 - margin) * 0.3 * (1 - N)

        S = np.minimum(remaining, s_duplicate + s_entropy + s_margin)

        # === VECTORIZED RELEVANT CALCULATION ===
        R = np.maximum(0.0, 1.0 - N - S)

        # === Normalize to ensure R + S + N = 1 ===
        total = R + S + N
        R = R / total
        S = S / total
        N = N / total

        # === Compute derived scores ===
        quality_score = R + 0.5 * S
        risk_score = S + 1.5 * N

        # === Determine collapse types ===
        collapse_types = np.where(
            (N > 0.3) & (S > 0.25), CollapseType.CONFUSION.value,
            np.where(
                N > 0.3, CollapseType.POISONING.value,
                np.where(
                    S > 0.4, CollapseType.DISTRACTION.value,
                    np.where(
                        risk_score > 0.6, CollapseType.CLASH.value,
                        CollapseType.NONE.value
                    )
                )
            )
        )

        # Build result DataFrame
        result_df = cleanlab_df.copy()
        result_df['R'] = R
        result_df['S'] = S
        result_df['N'] = N
        result_df['quality_score'] = quality_score
        result_df['risk_score'] = risk_score
        result_df['collapse_type'] = collapse_types

        return result_df

    # =========================================================================
    # DATALAB INTEGRATION (Cleanlab's unified interface)
    # =========================================================================

    def from_datalab(
        self,
        lab: Any,  # Datalab instance
        include_details: bool = False
    ) -> pd.DataFrame:
        """
        Convert Datalab results to YRSN.

        Datalab provides a unified interface for detecting multiple issue types.
        This method extracts all available signals and converts to YRSN.

        Parameters
        ----------
        lab : Datalab
            Cleanlab Datalab instance after running find_issues()
        include_details : bool
            If True, include all Datalab columns in output

        Returns
        -------
        pd.DataFrame
            DataFrame with YRSN columns for each example
        """
        # Get issues DataFrame from Datalab
        issues_df = lab.get_issues()

        # Map Datalab columns to our expected format
        column_mapping = {
            'label_score': 'label_quality',
            'is_label_issue': 'is_label_issue',
            'outlier_score': 'ood_score',
            'is_outlier_issue': 'is_ood',
            'near_duplicate_score': 'duplicate_score',
            'is_near_duplicate_issue': 'is_near_duplicate',
        }

        # Rename available columns
        renamed_df = issues_df.rename(columns={
            k: v for k, v in column_mapping.items() if k in issues_df.columns
        })

        # Ensure label_quality exists (invert is_label_issue if needed)
        if 'label_quality' not in renamed_df.columns:
            if 'label_score' in issues_df.columns:
                renamed_df['label_quality'] = issues_df['label_score']
            else:
                renamed_df['label_quality'] = 0.8  # Default

        # Invert ood_score if from outlier detection (Cleanlab: 1 = likely outlier)
        if 'ood_score' in renamed_df.columns:
            # Cleanlab outlier_score: higher = more likely outlier
            # We want: higher = more in-distribution
            renamed_df['ood_score'] = 1 - renamed_df['ood_score']

        # Use vectorized method for conversion
        yrsn_df = self.batch_to_yrsn_vectorized(
            renamed_df,
            label_quality_col='label_quality',
            ood_col='ood_score' if 'ood_score' in renamed_df.columns else None,
            duplicate_col='is_near_duplicate' if 'is_near_duplicate' in renamed_df.columns else None
        )

        if include_details:
            # Merge with original Datalab output
            return pd.concat([issues_df, yrsn_df[['R', 'S', 'N', 'quality_score', 'risk_score', 'collapse_type']]], axis=1)

        return yrsn_df

    # =========================================================================
    # MULTI-ANNOTATOR SUPPORT (for multi-approver scenarios)
    # =========================================================================

    def from_multiannotator(
        self,
        labels_multiannotator: np.ndarray,
        consensus_label: np.ndarray,
        annotator_agreement: np.ndarray,
        quality_of_consensus: np.ndarray,
        num_annotations: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Convert multi-annotator results to YRSN.

        For approval scenarios with multiple approvers, this method
        maps annotator agreement/disagreement to YRSN components.

        Parameters
        ----------
        labels_multiannotator : np.ndarray
            (N, M) array where M = max annotators, NaN for missing
        consensus_label : np.ndarray
            (N,) best consensus label from Cleanlab
        annotator_agreement : np.ndarray
            (N,) agreement score [0, 1] for each example
        quality_of_consensus : np.ndarray
            (N,) quality score for the consensus label [0, 1]
        num_annotations : np.ndarray, optional
            (N,) number of annotations per example

        Returns
        -------
        pd.DataFrame
            YRSN decomposition with multi-annotator context
        """
        n = len(consensus_label)

        # === Noise: Low consensus quality indicates potential labeling errors ===
        N = 1 - quality_of_consensus

        # === Superfluous: Low agreement but not wrong ===
        # Disagreement that doesn't indicate errors → ambiguous cases
        S = (1 - annotator_agreement) * quality_of_consensus

        # Boost S for examples with many conflicting annotations
        if num_annotations is not None:
            many_annotators = num_annotations >= 3
            S[many_annotators] *= 1.2

        S = np.clip(S, 0, 1 - N)

        # === Relevant: What remains ===
        R = np.maximum(0, 1 - N - S)

        # Normalize
        total = R + S + N
        R, S, N = R/total, S/total, N/total

        # Quality and risk
        quality_score = R + 0.5 * S
        risk_score = S + 1.5 * N

        # Collapse types
        collapse_types = []
        for r, s, n, risk in zip(R, S, N, risk_score):
            if n > 0.3 and s > 0.25:
                collapse_types.append(CollapseType.CONFUSION.value)
            elif n > 0.3:
                collapse_types.append(CollapseType.POISONING.value)
            elif s > 0.4:
                collapse_types.append(CollapseType.DISTRACTION.value)
            elif risk > 0.6:
                collapse_types.append(CollapseType.CLASH.value)
            else:
                collapse_types.append(CollapseType.NONE.value)

        return pd.DataFrame({
            'consensus_label': consensus_label,
            'annotator_agreement': annotator_agreement,
            'quality_of_consensus': quality_of_consensus,
            'R': R,
            'S': S,
            'N': N,
            'quality_score': quality_score,
            'risk_score': risk_score,
            'collapse_type': collapse_types
        })

    # =========================================================================
    # CLASS-LEVEL QUALITY ANALYSIS
    # =========================================================================

    def from_overlapping_classes(
        self,
        overlap_matrix: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get class-level confusion (N) from overlapping classes.

        When classes frequently get confused with each other, this indicates
        noise at the class level, not just individual examples.

        Parameters
        ----------
        overlap_matrix : np.ndarray
            (K, K) matrix of class overlap scores from Cleanlab
        class_names : list, optional
            Names for the K classes

        Returns
        -------
        dict
            Per-class YRSN breakdown: {class_name: {"R": r, "S": s, "N": n}}
        """
        K = overlap_matrix.shape[0]
        class_names = class_names or [f"class_{i}" for i in range(K)]

        result = {}
        for i in range(K):
            # N: How much this class overlaps with others (off-diagonal sum)
            overlap_with_others = (overlap_matrix[i, :].sum() - overlap_matrix[i, i])
            total_overlap = overlap_matrix[i, :].sum()

            if total_overlap > 0:
                N = overlap_with_others / total_overlap
            else:
                N = 0.0

            # S: Self-overlap that isn't clearly this class
            S = max(0, 0.3 - overlap_matrix[i, i]) if total_overlap > 0 else 0.0

            # R: Clean, clearly this class
            R = max(0, 1 - N - S)

            # Normalize
            total = R + S + N
            if total > 0:
                R, S, N = R/total, S/total, N/total
            else:
                R, S, N = 0.33, 0.33, 0.34

            result[class_names[i]] = {
                "R": R, "S": S, "N": N,
                "quality_score": R + 0.5 * S,
                "risk_score": S + 1.5 * N,
                "overlap_with_others": overlap_with_others
            }

        return result

    def from_class_quality(
        self,
        class_quality_df: pd.DataFrame,
        quality_col: str = "Label Quality",
        class_col: str = "Class Name"
    ) -> Dict[str, YRSNResult]:
        """
        Get per-class YRSN from class quality ranking.

        Uses rank_classes_by_label_quality() output to identify
        which classes have data quality issues.

        Parameters
        ----------
        class_quality_df : pd.DataFrame
            Output from cleanlab.dataset.rank_classes_by_label_quality()
        quality_col : str
            Column name for quality scores
        class_col : str
            Column name for class names/indices

        Returns
        -------
        dict
            {class_name: YRSNResult} for each class
        """
        result = {}

        for _, row in class_quality_df.iterrows():
            class_name = str(row[class_col])
            quality = row[quality_col]

            # Map quality to YRSN
            # High quality → high R
            # Low quality → high N (class-level noise)
            N = max(0, 1 - quality)
            S = max(0, (1 - quality) * 0.3)  # Some ambiguity
            R = max(0, 1 - N - S)

            # Normalize
            total = R + S + N
            result[class_name] = YRSNResult(R=R/total, S=S/total, N=N/total)

        return result

    # =========================================================================
    # DATASET HEALTH SCORE
    # =========================================================================

    def from_health_summary(
        self,
        health_score: float,
        label_issues_fraction: float,
        outlier_fraction: float = 0.0,
        duplicate_fraction: float = 0.0
    ) -> YRSNResult:
        """
        Convert overall dataset health to YRSN.

        Uses overall_label_health_score() and related metrics
        to get a dataset-level quality assessment.

        Parameters
        ----------
        health_score : float
            Overall label health score [0, 1] from Cleanlab
        label_issues_fraction : float
            Fraction of examples with label issues [0, 1]
        outlier_fraction : float
            Fraction of examples that are outliers [0, 1]
        duplicate_fraction : float
            Fraction of examples that are duplicates [0, 1]

        Returns
        -------
        YRSNResult
            Dataset-level YRSN decomposition
        """
        # N: Primary from label issues and outliers
        N = label_issues_fraction + outlier_fraction * 0.5
        N = min(N, 1.0)

        # S: Duplicates and portion of borderline cases
        S = duplicate_fraction * self.duplicate_weight
        S = min(S, 1.0 - N)

        # R: Health score contribution
        R = health_score * (1 - N - S) + (1 - N - S) * (1 - health_score) * 0.5
        R = max(0, min(R, 1 - N - S))

        # Adjust to sum to 1
        remaining = 1 - N - S - R
        if remaining > 0:
            R += remaining

        return YRSNResult(R=R, S=S, N=N)

    # =========================================================================
    # CONFIDENT THRESHOLDS (Per-class reliability)
    # =========================================================================

    def from_confident_thresholds(
        self,
        thresholds: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze per-class decision reliability from confident thresholds.

        Classes with very different thresholds from the mean indicate
        either very clean (low threshold) or very noisy (high threshold) data.

        Parameters
        ----------
        thresholds : np.ndarray
            (K,) array of confident thresholds from get_confident_thresholds()
        class_names : list, optional
            Names for the K classes

        Returns
        -------
        dict
            Per-class reliability metrics
        """
        K = len(thresholds)
        class_names = class_names or [f"class_{i}" for i in range(K)]

        mean_threshold = np.mean(thresholds)
        std_threshold = np.std(thresholds)

        result = {}
        for i, (name, thresh) in enumerate(zip(class_names, thresholds)):
            # Deviation from mean (normalized)
            if std_threshold > 0:
                z_score = (thresh - mean_threshold) / std_threshold
            else:
                z_score = 0

            # High threshold = needs more confidence = noisier class
            # Low threshold = accepts lower confidence = cleaner class
            if thresh > mean_threshold + std_threshold:
                # Noisy class: high N
                N = min(0.5, 0.3 + 0.1 * z_score)
                S = 0.2
                R = 1 - N - S
            elif thresh < mean_threshold - std_threshold:
                # Clean class: high R
                R = min(0.9, 0.7 + 0.1 * abs(z_score))
                N = 0.05
                S = 1 - R - N
            else:
                # Average class
                R = 0.6
                S = 0.2
                N = 0.2

            result[name] = {
                "threshold": thresh,
                "z_score": z_score,
                "R": R, "S": S, "N": N,
                "reliability": "high" if R > 0.7 else "medium" if R > 0.5 else "low"
            }

        return result
