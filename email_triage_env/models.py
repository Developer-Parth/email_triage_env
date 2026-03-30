"""
Pydantic models for the Email Triage OpenEnv Environment.
"""

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EmailCategory(str, Enum):
    """Valid email classification categories."""
    SPAM = "spam"
    WORK = "work"
    PERSONAL = "personal"


class PriorityLevel(str, Enum):
    """Valid priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Observation(BaseModel):
    """
    Observation visible to the agent at each step.
    
    The agent sees email content but not ground truth labels.
    """
    email_id: str = Field(description="Unique identifier for the email")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    step_count: int = Field(description="Current step number in the episode")
    history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional history of previous actions taken"
    )
    
    class Config:
        frozen = True  # Immutable observations


class Action(BaseModel):
    """
    Structured action that the agent can take.
    
    The agent can classify, prioritize, or reply to the email.
    """
    action_type: Literal["classify", "prioritize", "reply"] = Field(
        description="Type of action to perform"
    )
    
    # Action-specific parameters
    label: Optional[EmailCategory] = Field(
        default=None,
        description="Classification label (required for 'classify' action)"
    )
    level: Optional[PriorityLevel] = Field(
        default=None,
        description="Priority level (required for 'prioritize' action)"
    )
    text: Optional[str] = Field(
        default=None,
        description="Reply text (required for 'reply' action)"
    )
    
    class Config:
        frozen = True
    
    def validate_action(self):
        """Validate that action has appropriate parameters."""
        if self.action_type == "classify" and self.label is None:
            raise ValueError("Classification action requires a label")
        if self.action_type == "prioritize" and self.level is None:
            raise ValueError("Prioritization action requires a level")
        if self.action_type == "reply" and self.text is None:
            raise ValueError("Reply action requires text")
        return True


class Reward(BaseModel):
    """
    Reward signal returned after each action.
    
    Provides detailed feedback on agent performance.
    """
    total: float = Field(description="Total reward for this step")
    classification: Optional[float] = Field(
        default=None,
        description="Reward for classification (if applicable)"
    )
    priority: Optional[float] = Field(
        default=None,
        description="Reward for prioritization (if applicable)"
    )
    sequence: Optional[float] = Field(
        default=None,
        description="Reward for action sequence correctness"
    )
    invalid_penalty: Optional[float] = Field(
        default=None,
        description="Penalty for invalid actions"
    )
    completion_bonus: Optional[float] = Field(
        default=None,
        description="Bonus for completing all required steps"
    )
    step_cost: Optional[float] = Field(
        default=None,
        description="Cost per step to encourage efficiency"
    )
    state_traps: Optional[float] = Field(
        default=None,
        description="Penalty from state-based traps (changing correct decisions, repeating actions, etc.)"
    )
    
    class Config:
        frozen = True


class State(BaseModel):
    """
    Internal state of the environment (not exposed to agent).
    
    Contains ground truth and progress tracking.
    """
    email_id: str
    subject: str
    body: str
    ground_truth_category: EmailCategory
    ground_truth_priority: PriorityLevel
    is_classified: bool = False
    is_prioritized: bool = False
    is_replied: bool = False
    step_count: int = 0
    max_steps: int = 8
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    task_type: Literal["easy", "medium", "hard"] = "easy"
    
    # State-based trap tracking
    current_classification: Optional[EmailCategory] = Field(default=None)
    current_priority: Optional[PriorityLevel] = Field(default=None)
    last_action_type: Optional[str] = Field(default=None)
    classification_changed_after_correct: bool = Field(default=False)
    priority_changed_after_correct: bool = Field(default=False)
    repeated_action_count: Dict[str, int] = Field(default_factory=lambda: {"classify": 0, "prioritize": 0, "reply": 0})
    
    class Config:
        frozen = False  # State is mutable