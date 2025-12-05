"""
YRSN-IARS: Intelligent Approval Routing System

Combines Cleanlab data quality with YRSN context decomposition
for intelligent approval routing with confidence scoring.
"""

__version__ = "0.1.0"

from .adapters.cleanlab_adapter import CleanlabAdapter, YRSNResult
from .pipelines.approval_router import ApprovalRouter, RoutingDecision

__all__ = [
    "CleanlabAdapter",
    "YRSNResult",
    "ApprovalRouter",
    "RoutingDecision",
]
