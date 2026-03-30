"""
Task definitions and graders for Email Triage environment.

Contains three tasks with increasing difficulty:
1. Easy Task - Classification only
2. Medium Task - Prioritization only  
3. Hard Task - Full triage (classification + prioritization + optional reply)
"""

from .graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader

__all__ = ["EasyTaskGrader", "MediumTaskGrader", "HardTaskGrader"]