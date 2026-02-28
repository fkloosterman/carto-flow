"""Status enum for symbol cartogram operations."""

from __future__ import annotations

from enum import Enum


class SymbolCartogramStatus(str, Enum):
    """Status values for symbol cartogram operations.

    Attributes
    ----------
    CONVERGED
        Simulation converged: no overlaps remaining and velocities below threshold.
    COMPLETED
        Maximum iterations reached without full convergence.
    ORIGINAL
        No processing done (e.g., single geometry input).

    """

    CONVERGED = "converged"
    COMPLETED = "completed"
    ORIGINAL = "original"
