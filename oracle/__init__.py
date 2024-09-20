"""
Observation, Research, and Analysis of Collapse and Loading Experiments (ORACLE).

This package implements closed-form analytical models for the analysis of
anticracks in the avalanche release process.
"""

# Relative imports of classes
from .snowpilot import SnowPilotQueryEngine, SnowPilotParser
from .slf import SLFDataParser
from .pst import PropagationSawTestEngine
from . import plot

# Version
__version__ = '0.1.0'

# Public API
__all__ = [
    'SnowPilotQueryEngine',
    'SnowPilotParser',
    'SLFDataParser',
    'PropagationSawTestEngine',
    'plot',
]