# trading_engine/__init__.py
"""
Trading Engine Package
Handles:
- Indicator calculations
- Core trading logic (entries, exits, risk)
- Parameter optimization
"""

from .engine import TradingEngine
from .indicators import calculate_indicators
from .optimizer import optimize_parameters

__all__ = [
    "TradingEngine",
    "calculate_indicators",
    "optimize_parameters",
]
