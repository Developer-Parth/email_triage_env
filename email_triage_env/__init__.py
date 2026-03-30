"""
Email Triage OpenEnv Environment

A real-world simulation for training AI agents on handling incoming emails.
The agent must read email content, classify it, assign priority, and optionally generate responses.
"""

from .environment import EmailTriageEnv
from .models import Observation, Action, Reward, State
from .tasks.graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader

__version__ = "0.1.0"
__all__ = [
    "EmailTriageEnv",
    "Observation",
    "Action",
    "Reward",
    "State",
    "EasyTaskGrader",
    "MediumTaskGrader",
    "HardTaskGrader",
]
