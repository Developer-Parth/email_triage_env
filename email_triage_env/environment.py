"""
Email Triage OpenEnv Environment implementation.

Implements the core environment with state management and OpenEnv APIs.
"""

import random
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from .models import (
    Observation, Action, Reward, State,
    EmailCategory, PriorityLevel
)

# Scoring stability constants
FLOAT_TOLERANCE = 1e-9
REWARD_PRECISION = 6  # Decimal places for rounding rewards


def round_reward(value: Optional[float]) -> Optional[float]:
    """Round reward value to ensure scoring stability."""
    if value is None:
        return None
    # Use Python's round with decimal precision
    return round(value, REWARD_PRECISION)


def safe_float_eq(a: float, b: float) -> bool:
    """Compare floats with tolerance for scoring stability."""
    return abs(a - b) < FLOAT_TOLERANCE


def safe_float_gt(a: float, b: float) -> bool:
    """Check if a > b with tolerance."""
    return a - b > FLOAT_TOLERANCE


def safe_float_gte(a: float, b: float) -> bool:
    """Check if a >= b with tolerance."""
    return a - b > -FLOAT_TOLERANCE


@dataclass
class EmailDataset:
    """Dataset of emails with ground truth labels and train/test split support."""
    emails: List[Dict[str, Any]] = field(default_factory=list)
    test_emails: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "train"  # "train", "test", or "mixed"
    
    def __post_init__(self):
        """Initialize with sample emails if empty."""
        if not self.emails and not self.test_emails:
            train_emails, test_emails = self._create_split_sample_emails()
            self.emails = train_emails
            self.test_emails = test_emails
    
    def _create_split_sample_emails(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create a diverse set of sample emails split into train and test sets."""
        all_emails = [
            # Clear-cut emails (easy) - TRAIN
            {
                "id": "email_001",
                "subject": "Urgent: Project Deadline Tomorrow",
                "body": "Hi team, the project deadline is tomorrow. Please submit your final reports by 5 PM. This is critical for client delivery.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.HIGH,
                "difficulty": "easy",
                "split": "train"
            },
            {
                "id": "email_002",
                "subject": "Weekend BBQ at My Place",
                "body": "Hey everyone! I'm hosting a BBQ this Saturday at 3 PM. Bring your favorite drinks and snacks. Let me know if you can make it!",
                "category": EmailCategory.PERSONAL,
                "priority": PriorityLevel.LOW,
                "difficulty": "easy",
                "split": "train"
            },
            {
                "id": "email_003",
                "subject": "WIN A FREE IPHONE!!!",
                "body": "CONGRATULATIONS! You have been selected to win a FREE iPhone. Click here to claim your prize now!",
                "category": EmailCategory.SPAM,
                "priority": PriorityLevel.LOW,
                "difficulty": "easy",
                "split": "train"
            },
            
            # Ambiguous emails (medium difficulty) - TRAIN
            {
                "id": "email_004",
                "subject": "Reminder about tomorrow",
                "body": "Just a quick reminder about our plans for tomorrow. Don't forget to bring the documents we discussed.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.MEDIUM,
                "difficulty": "medium",
                "split": "train"
            },
            {
                "id": "email_005",
                "subject": "Your subscription renewal",
                "body": "Your premium subscription will renew automatically next week. This is a transactional email about your account.",
                "category": EmailCategory.PERSONAL,
                "priority": PriorityLevel.MEDIUM,
                "difficulty": "medium",
                "split": "train"
            },
            
            # Challenging emails (hard difficulty) - TRAIN
            {
                "id": "email_006",
                "subject": "Important: Action required on your account",
                "body": "We've detected suspicious activity. Please verify your identity immediately to prevent account suspension.",
                "category": EmailCategory.PERSONAL,
                "priority": PriorityLevel.HIGH,
                "difficulty": "hard",
                "split": "train"
            },
            {
                "id": "email_007",
                "subject": "Project feedback request",
                "body": "Hi, could you please review the attached document when you have a moment? No rush, just whenever you're free.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.LOW,
                "difficulty": "hard",
                "split": "train"
            },
            
            # Noise/mixed signal emails (very hard) - TRAIN
            {
                "id": "email_008",
                "subject": "URGENT but not really",
                "body": "This is marked urgent but actually it's just a routine update. The system automatically flagged it as high priority.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.LOW,
                "difficulty": "very_hard",
                "split": "train"
            },
            
            # TEST SET EMAILS (unseen during training)
            {
                "id": "email_test_001",
                "subject": "Company social event Friday",
                "body": "Join us for the company social event this Friday at 6 PM. Attendance is optional but encouraged for team building.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.LOW,
                "difficulty": "medium",
                "split": "test"
            },
            {
                "id": "email_test_002",
                "subject": "Limited time offer exclusive for you",
                "body": "As a valued customer, we're offering you an exclusive discount on our premium services. This offer expires in 24 hours.",
                "category": EmailCategory.SPAM,
                "priority": PriorityLevel.MEDIUM,
                "difficulty": "hard",
                "split": "test"
            },
            {
                "id": "email_test_003",
                "subject": "Friendly reminder from HR",
                "body": "This is a friendly reminder about the upcoming deadline for benefits enrollment. Please ignore if you've already completed.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.MEDIUM,
                "difficulty": "very_hard",
                "split": "test"
            },
            {
                "id": "email_test_004",
                "subject": "Your package delivery failed",
                "body": "We attempted to deliver your package but no one was home. This is your final attempt before return to sender.",
                "category": EmailCategory.PERSONAL,
                "priority": PriorityLevel.HIGH,
                "difficulty": "very_hard",
                "split": "test"
            },
            {
                "id": "email_test_005",
                "subject": "IMPORTANT: Please disregard",
                "body": "This email was marked as important by mistake. Please disregard its contents. It contains no actionable information.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.LOW,
                "difficulty": "very_hard",
                "split": "test"
            },
            {
                "id": "email_test_006",
                "subject": "Casual Friday reminder",
                "body": "REMINDER: Tomorrow is casual Friday. This is a MANDATORY dress code policy. Failure to comply may result in disciplinary action.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.MEDIUM,
                "difficulty": "hard",
                "split": "test"
            },
            {
                "id": "email_test_007",
                "subject": "Free lunch tomorrow!",
                "body": "The company is providing free lunch for all employees in the cafeteria. Attendance is completely optional with no strings attached.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.LOW,
                "difficulty": "medium",
                "split": "test"
            },
            {
                "id": "email_test_008",
                "subject": "Meeting cancellation",
                "body": "The 3 PM meeting has been CANCELLED. This is URGENT information that requires immediate attention to adjust your schedule.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.HIGH,
                "difficulty": "medium",
                "split": "test"
            },
            
            # INSANE EDGE CASES - Realistic tricky emails that require deep reasoning
            # 1. Email that looks like spam but is actually legitimate (phishing vs legit security alert)
            {
                "id": "email_edge_001",
                "subject": "SECURITY ALERT: Unusual login detected from New York",
                "body": "We detected a login to your account from an unrecognized device in New York at 3:47 AM. If this was you, no action is needed. If not, please secure your account immediately by clicking here: https://secure-bank.example.com/verify",
                "category": EmailCategory.PERSONAL,  # Legitimate security alert (personal finance)
                "priority": PriorityLevel.HIGH,  # High priority if legitimate
                "difficulty": "very_hard",
                "split": "train",  # Put in train set to teach agents this pattern
                "reasoning_hint": "Looks like phishing but could be legitimate security alert. Requires checking sender domain and context."
            },
            
            # 2. Work email with casual/personal tone (blurred boundaries)
            {
                "id": "email_edge_002",
                "subject": "Hey, got a minute?",
                "body": "Hey buddy, just wanted to quickly chat about that thing we discussed last week over beers. Nothing urgent, but would be great to sync when you're free. No pressure though! Cheers, Mike",
                "category": EmailCategory.WORK,  # From colleague about work matter
                "priority": PriorityLevel.LOW,  # Casual tone suggests low priority
                "difficulty": "very_hard",
                "split": "test",  # Put in test set to test generalization
                "reasoning_hint": "Casual tone but from work colleague about work topic. Requires understanding professional relationships."
            },
            
            # 3. Urgent but non-critical message with mixed signals
            {
                "id": "email_edge_003",
                "subject": "TIME-SENSITIVE: Quarterly planning document",
                "body": "The quarterly planning document needs your review by EOD today. However, this is just for alignment purposes and won't block any critical paths if delayed until tomorrow.",
                "category": EmailCategory.WORK,
                "priority": PriorityLevel.MEDIUM,  # Time-sensitive but not critical
                "difficulty": "very_hard",
                "split": "train",  # Put in train set
                "reasoning_hint": "Marked time-sensitive but content says it's not critical. Requires parsing contradictory signals."
            },
            
            # 4. Personal email that requires work action (cross-boundary)
            {
                "id": "email_edge_004",
                "subject": "Your health insurance renewal",
                "body": "Your employee health insurance plan is up for renewal. This requires action through the HR portal by Friday. Failure to renew will result in loss of coverage.",
                "category": EmailCategory.PERSONAL,  # Personal benefit but through work
                "priority": PriorityLevel.HIGH,  # High priority due to consequences
                "difficulty": "very_hard",
                "split": "test",  # Put in test set
                "reasoning_hint": "Personal matter (health insurance) but requires work system action. Tests boundary understanding."
            }
        ]
        
        # Split into train and test
        train_emails = [e for e in all_emails if e["split"] == "train"]
        test_emails = [e for e in all_emails if e["split"] == "test"]
        
        return train_emails, test_emails
    
    def get_random_email(self) -> Dict[str, Any]:
        """Return a random email based on current mode."""
        if self.mode == "train":
            return random.choice(self.emails)
        elif self.mode == "test":
            return random.choice(self.test_emails)
        else:  # mixed
            return random.choice(self.emails + self.test_emails)
    
    def set_mode(self, mode: str) -> None:
        """Set the dataset mode: 'train', 'test', or 'mixed'."""
        if mode not in ["train", "test", "mixed"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'mixed'.")
        self.mode = mode
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        return {
            "train_emails": len(self.emails),
            "test_emails": len(self.test_emails),
            "total_emails": len(self.emails) + len(self.test_emails),
            "mode": self.mode,
            "train_categories": self._count_categories(self.emails),
            "test_categories": self._count_categories(self.test_emails)
        }
    
    def _count_categories(self, email_list: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count categories in an email list."""
        counts = {}
        for email in email_list:
            cat = email["category"].value
            counts[cat] = counts.get(cat, 0) + 1
        return counts


class EmailTriageEnv:
    """
    Email Triage OpenEnv Environment.
    
    Simulates email triage task with classification, prioritization, and reply generation.
    """
    
    def __init__(self, task_type: str = "easy", max_steps: int = 8, dataset_mode: str = "train"):
        """
        Initialize the environment.
        
        Args:
            task_type: "easy", "medium", or "hard"
            max_steps: Maximum steps per episode
            dataset_mode: "train", "test", or "mixed" - controls which emails are used
        """
        self.task_type = task_type
        self.max_steps = max_steps
        self.dataset = EmailDataset()
        self.dataset.set_mode(dataset_mode)
        self._state: Optional[State] = None
        self._current_email: Optional[Dict[str, Any]] = None
        
    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Initial observation
        """
        if seed is not None:
            random.seed(seed)
        
        # Select a random email
        self._current_email = self.dataset.get_random_email()
        
        # Initialize state
        self._state = State(
            email_id=self._current_email["id"],
            subject=self._current_email["subject"],
            body=self._current_email["body"],
            ground_truth_category=self._current_email["category"],
            ground_truth_priority=self._current_email["priority"],
            step_count=0,
            max_steps=self.max_steps,
            task_type=self.task_type,
            action_history=[]
        )
        
        # Return initial observation
        return self._get_observation()
    
    def _get_observation(self) -> Observation:
        """Create observation from current state."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        return Observation(
            email_id=self._state.email_id,
            subject=self._state.subject,
            body=self._state.body,
            step_count=self._state.step_count,
            history=self._state.action_history[-3:] if self._state.action_history else None
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute an action and return the result.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Validate action
        try:
            action.validate_action()
        except ValueError as e:
            # Invalid action - penalize and return same observation
            reward = Reward(
                total=-0.2,
                invalid_penalty=-0.2
            )
            done = self._check_episode_end()
            return self._get_observation(), reward, done, {"error": str(e)}
        
        # Update step count
        self._state.step_count += 1
        
        # Record action in history
        action_record = {
            "step": self._state.step_count,
            "action_type": action.action_type,
            "label": action.label,
            "level": action.level,
            "text": action.text[:50] + "..." if action.text and len(action.text) > 50 else action.text
        }
        self._state.action_history.append(action_record)
        
        # Initialize reward components
        reward_components = {}
        
        # Check for state-based traps and apply penalties
        trap_penalties = self._check_state_traps(action)
        if trap_penalties:
            reward_components["state_traps"] = trap_penalties
        
        # Process action based on type
        if action.action_type == "classify":
            reward_components["classification"] = self._evaluate_classification(action.label)
            if reward_components["classification"] > 0:
                self._state.is_classified = True
            # Update current classification for tracking
            self._state.current_classification = action.label
        
        elif action.action_type == "prioritize":
            reward_components["priority"] = self._evaluate_priority(action.level)
            if reward_components["priority"] > 0:
                self._state.is_prioritized = True
            # Update current priority for tracking
            self._state.current_priority = action.level
        
        elif action.action_type == "reply":
            # For hard task only
            if self.task_type == "hard":
                reward_components["reply"] = self._evaluate_reply(action.text)
                if reward_components["reply"] is not None:
                    self._state.is_replied = True
        
        # Calculate sequence reward
        reward_components["sequence"] = self._evaluate_sequence(action)
        
        # Add step cost to encourage efficiency (-0.05 per step)
        step_cost = round_reward(-0.05) or -0.05
        reward_components["step_cost"] = step_cost
        
        # Calculate total reward with rounding for stability
        total_reward = sum(v for v in reward_components.values() if v is not None)
        total_reward = round_reward(total_reward) or total_reward
        
        # Add completion bonus if episode is ending successfully
        done = self._check_episode_end()
        completion_bonus = 0.0
        if done and self._is_task_completed():
            completion_bonus = round_reward(0.3) or 0.3  # Increased to offset step costs
            reward_components["completion_bonus"] = completion_bonus
            total_reward += completion_bonus
            total_reward = round_reward(total_reward) or total_reward
        
        # Update last action type for tracking repeats
        self._state.last_action_type = action.action_type
        
        # Create reward object with rounded values
        reward = Reward(
            total=total_reward,
            classification=round_reward(reward_components.get("classification")),
            priority=round_reward(reward_components.get("priority")),
            sequence=round_reward(reward_components.get("sequence")),
            step_cost=step_cost,
            completion_bonus=round_reward(completion_bonus) if completion_bonus > 0 else None,
            state_traps=round_reward(reward_components.get("state_traps"))
        )
        
        # Prepare detailed info dict with evaluation logs
        info = {
            "step": self._state.step_count,
            "task_type": self.task_type,
            "is_classified": self._state.is_classified,
            "is_prioritized": self._state.is_prioritized,
            "is_replied": self._state.is_replied if self.task_type == "hard" else None,
            "action_history": self._state.action_history,
            "trap_details": action_record.get("state_traps", []),
            # Detailed evaluation logs
            "evaluation": {
                "email_id": self._state.email_id,
                "email_difficulty": self._current_email.get("difficulty", "unknown") if self._current_email else "unknown",
                "ground_truth": {
                    "category": self._state.ground_truth_category.value,
                    "priority": self._state.ground_truth_priority.value
                },
                "action_type": action.action_type,
                "action_details": {
                    "predicted_category": action.label.value if action.label else None,
                    "predicted_priority": action.level.value if action.level else None,
                    "reply_length": len(action.text) if action.text else 0
                },
                "reward_breakdown": reward_components,
                "sequence_correctness": self._get_sequence_correctness_details(action),
                "total_reward": total_reward,
                "step_limit_warning": self._state.step_count >= self._state.max_steps - 2
            }
        }
        
        return self._get_observation(), reward, done, info
    
    def _evaluate_classification(self, predicted_label: EmailCategory) -> float:
        """
        Evaluate classification action with anti-exploitation measures.
        
        Uses asymmetric rewards: penalties are larger than rewards to discourage
        random guessing and exploitation strategies.
        """
        if self._state is None or self._current_email is None:
            return round_reward(0.0) or 0.0
        
        difficulty = self._current_email.get("difficulty", "medium")
        
        if predicted_label == self._state.ground_truth_category:
            # Correct classification - modest reward
            if difficulty == "easy":
                return round_reward(0.4) or 0.4  # Small reward for easy emails
            elif difficulty == "medium":
                return round_reward(0.5) or 0.5
            elif difficulty == "hard":
                return round_reward(0.6) or 0.6  # Moderate reward for hard emails
            elif difficulty == "very_hard":
                return round_reward(0.7) or 0.7  # Good reward for very hard emails
            else:
                return round_reward(0.5) or 0.5
        else:
            # Incorrect classification - significant penalty
            # Penalties are 1.5-2x larger than rewards to discourage guessing
            if difficulty == "easy":
                return round_reward(-0.8) or -0.8  # Heavy penalty for wrong on easy email
            elif difficulty == "medium":
                return round_reward(-0.7) or -0.7
            elif difficulty == "hard":
                return round_reward(-0.6) or -0.6  # Still significant penalty for hard emails
            elif difficulty == "very_hard":
                return round_reward(-0.5) or -0.5  # Moderate penalty for very hard emails
            else:
                return round_reward(-0.6) or -0.6
    
    def _evaluate_priority(self, predicted_level: PriorityLevel) -> float:
        """
        Evaluate prioritization action with anti-exploitation measures.
        
        Penalties exceed rewards to prevent always-guessing strategies.
        """
        if self._state is None or self._current_email is None:
            return round_reward(0.0) or 0.0
        
        difficulty = self._current_email.get("difficulty", "medium")
        
        if predicted_level == self._state.ground_truth_priority:
            # Correct priority - modest reward
            if difficulty == "easy":
                return round_reward(0.2) or 0.2  # Small reward for easy priority
            elif difficulty == "medium":
                return round_reward(0.25) or 0.25
            elif difficulty == "hard":
                return round_reward(0.3) or 0.3  # Moderate reward for hard priority
            elif difficulty == "very_hard":
                return round_reward(0.35) or 0.35  # Good reward for very hard priority
            else:
                return round_reward(0.25) or 0.25
        else:
            # Incorrect priority - significant penalty
            if difficulty == "easy":
                return round_reward(-0.5) or -0.5  # Heavy penalty for wrong on easy priority
            elif difficulty == "medium":
                return round_reward(-0.45) or -0.45
            elif difficulty == "hard":
                return round_reward(-0.4) or -0.4  # Significant penalty for hard priority
            elif difficulty == "very_hard":
                return round_reward(-0.35) or -0.35  # Moderate penalty for very hard priority
            else:
                return round_reward(-0.4) or -0.4
    
    def _evaluate_reply(self, reply_text: str) -> Optional[float]:
        """Evaluate reply quality (simplified)."""
        if not reply_text:
            return None
        
        # Simple evaluation: check if reply is reasonable length
        # In a real implementation, this would use semantic similarity
        if len(reply_text.strip()) > 10:
            return round_reward(0.1) or 0.1  # Basic reply reward
        return round_reward(-0.1) or -0.1  # Too short reply
    
    def _evaluate_sequence(self, action: Action) -> float:
        """
        Evaluate action sequence correctness with strong dependency enforcement.
        
        Returns:
            Positive reward for correct sequence, negative penalty for violations
        """
        if self._state is None:
            return round_reward(0.0) or 0.0
        
        # Track sequence violations
        sequence_reward = 0.0
        
        # For easy task: classification must be first action
        if self.task_type == "easy":
            if self._state.step_count == 1:  # First action
                if action.action_type == "classify":
                    sequence_reward += round_reward(0.15) or 0.15  # Bonus for correct first action
                else:
                    sequence_reward -= round_reward(0.25) or 0.25  # Penalty for wrong first action
            # After classification, other actions are allowed but not required
            if action.action_type == "classify" and self._state.is_classified:
                sequence_reward -= round_reward(0.2) or 0.2  # Penalty for redundant classification
        
        # For medium task: prioritization must be first action
        elif self.task_type == "medium":
            if self._state.step_count == 1:  # First action
                if action.action_type == "prioritize":
                    sequence_reward += round_reward(0.15) or 0.15  # Bonus for correct first action
                else:
                    sequence_reward -= round_reward(0.25) or 0.25  # Penalty for wrong first action
            # After prioritization, other actions are allowed
            if action.action_type == "prioritize" and self._state.is_prioritized:
                sequence_reward -= round_reward(0.2) or 0.2  # Penalty for redundant prioritization
        
        # For hard task: strict dependency classification -> prioritization -> reply (optional)
        elif self.task_type == "hard":
            # Check for dependency violations
            if action.action_type == "prioritize" and not self._state.is_classified:
                # Trying to prioritize before classifying - major violation
                sequence_reward -= round_reward(0.4) or 0.4
                
            if action.action_type == "reply":
                # Reply should ideally come after classification and prioritization
                if not self._state.is_classified:
                    sequence_reward -= round_reward(0.3) or 0.3  # Replying before classification
                elif not self._state.is_prioritized:
                    sequence_reward -= round_reward(0.2) or 0.2  # Replying before prioritization
                else:
                    sequence_reward += round_reward(0.1) or 0.1  # Good sequence for reply
            
            # Check for redundant actions
            if action.action_type == "classify" and self._state.is_classified:
                sequence_reward -= round_reward(0.25) or 0.25  # Redundant classification
            
            if action.action_type == "prioritize" and self._state.is_prioritized:
                sequence_reward -= round_reward(0.25) or 0.25  # Redundant prioritization
            
            # Bonus for correct sequence progression
            if action.action_type == "classify" and not self._state.is_classified:
                sequence_reward += round_reward(0.1) or 0.1  # Good first step
            
            if (action.action_type == "prioritize" and self._state.is_classified
                    and not self._state.is_prioritized):
                sequence_reward += round_reward(0.15) or 0.15  # Good second step after classification
        
        return round_reward(sequence_reward) or sequence_reward
    
    def _check_state_traps(self, action: Action) -> float:
        """
        Check for state-based traps and return penalty if any.
        
        Traps include:
        1. Changing classification after correct classification
        2. Changing priority after correct prioritization
        3. Repeating same action type multiple times
        4. Inconsistent behavior patterns
        
        Returns:
            Total penalty (negative value) from all triggered traps
        """
        if self._state is None:
            return 0.0
        
        total_penalty = 0.0
        trap_details = []
        
        # 1. Check for changing classification after correct classification
        if (action.action_type == "classify" and
            self._state.current_classification is not None and
            self._state.is_classified and
            action.label != self._state.current_classification):
            # Agent is changing a previously correct classification - gentle nudge
            penalty = round_reward(-0.1) or -0.1  # Reduced from -0.3
            total_penalty += penalty
            trap_details.append({
                "trap": "changed_correct_classification",
                "penalty": penalty,
                "previous": self._state.current_classification.value,
                "new": action.label.value if action.label else None,
                "message": "Changing correct classification shows uncertainty"
            })
            self._state.classification_changed_after_correct = True
        
        # 2. Check for changing priority after correct prioritization
        if (action.action_type == "prioritize" and
            self._state.current_priority is not None and
            self._state.is_prioritized and
            action.level != self._state.current_priority):
            # Agent is changing a previously correct priority - gentle nudge
            penalty = round_reward(-0.05) or -0.05  # Reduced from -0.2
            total_penalty += penalty
            trap_details.append({
                "trap": "changed_correct_priority",
                "penalty": penalty,
                "previous": self._state.current_priority.value,
                "new": action.level.value if action.level else None,
                "message": "Changing correct priority shows inconsistency"
            })
            self._state.priority_changed_after_correct = True
        
        # 3. Check for repeating same action type
        if self._state.last_action_type == action.action_type:
            # Increment repeat count
            self._state.repeated_action_count[action.action_type] = \
                self._state.repeated_action_count.get(action.action_type, 0) + 1
            
            # Apply gentle penalty based on repeat count
            repeat_count = self._state.repeated_action_count[action.action_type]
            if repeat_count >= 3:  # Increased threshold from 2 to 3
                # Gentle penalty that increases slowly
                penalty = round_reward(-0.03 * (repeat_count - 2)) or (-0.03 * (repeat_count - 2))  # Reduced from -0.1
                total_penalty += penalty
                trap_details.append({
                    "trap": "repeated_action",
                    "penalty": penalty,
                    "action_type": action.action_type,
                    "repeat_count": repeat_count,
                    "message": f"Repeating {action.action_type} action {repeat_count} times shows inefficiency"
                })
        
        # 4. Check for inconsistent classification-priority patterns
        # (e.g., classifying as spam but prioritizing as high - contradictory)
        if (action.action_type == "prioritize" and
            self._state.current_classification is not None and
            action.level is not None):
            
            # Spam emails should generally be low priority
            if (self._state.current_classification == EmailCategory.SPAM and
                action.level == PriorityLevel.HIGH):
                penalty = round_reward(-0.15) or -0.15
                total_penalty += penalty
                trap_details.append({
                    "trap": "contradictory_spam_high_priority",
                    "penalty": penalty,
                    "classification": "spam",
                    "priority": "high"
                })
            
            # Work emails with urgent keywords should not be low priority
            if (self._state.current_classification == EmailCategory.WORK and
                action.level == PriorityLevel.LOW and
                any(keyword in self._state.body.lower() for keyword in ["urgent", "asap", "deadline"])):
                penalty = round_reward(-0.1) or -0.1
                total_penalty += penalty
                trap_details.append({
                    "trap": "work_urgent_low_priority",
                    "penalty": penalty,
                    "classification": "work",
                    "priority": "low",
                    "context": "email contains urgent keywords"
                })
        
        # Store trap details in state for debugging
        if trap_details:
            # Add to action history for debugging
            if self._state.action_history:
                self._state.action_history[-1]["state_traps"] = trap_details
        
        return round_reward(total_penalty) or total_penalty
    
    def _get_sequence_correctness_details(self, action: Action) -> Dict[str, Any]:
        """
        Generate detailed sequence correctness analysis for info logs.
        
        Returns:
            Dictionary with sequence analysis details
        """
        if self._state is None:
            return {"error": "state_not_initialized"}
        
        details = {
            "action_type": action.action_type,
            "step_number": self._state.step_count,
            "current_progress": {
                "is_classified": self._state.is_classified,
                "is_prioritized": self._state.is_prioritized,
                "is_replied": self._state.is_replied
            },
            "dependency_violations": [],
            "sequence_quality": "good"
        }
        
        # Check for dependency violations
        if self.task_type == "hard":
            if action.action_type == "prioritize" and not self._state.is_classified:
                details["dependency_violations"].append("prioritize_before_classify")
                details["sequence_quality"] = "poor"
            elif action.action_type == "reply":
                if not self._state.is_classified:
                    details["dependency_violations"].append("reply_before_classify")
                    details["sequence_quality"] = "poor"
                elif not self._state.is_prioritized:
                    details["dependency_violations"].append("reply_before_prioritize")
                    details["sequence_quality"] = "fair"
        
        # Check for redundant actions
        if action.action_type == "classify" and self._state.is_classified:
            details["dependency_violations"].append("redundant_classification")
            details["sequence_quality"] = "poor"
        elif action.action_type == "prioritize" and self._state.is_prioritized:
            details["dependency_violations"].append("redundant_prioritization")
            details["sequence_quality"] = "poor"
        
        # Check for correct first action
        if self._state.step_count == 1:
            if self.task_type == "easy" and action.action_type == "classify":
                details["first_action_correct"] = True
            elif self.task_type == "medium" and action.action_type == "prioritize":
                details["first_action_correct"] = True
            elif self.task_type == "hard":
                details["first_action_correct"] = (action.action_type == "classify")
            else:
                details["first_action_correct"] = False
        
        return details
    
    def _check_episode_end(self) -> bool:
        """Check if episode should end."""
        if self._state is None:
            return True
        
        # Check step limit
        if self._state.step_count >= self._state.max_steps:
            return True
        
        # Check task completion
        if self._is_task_completed():
            return True
        
        return False
    
    def _is_task_completed(self) -> bool:
        """Check if current task is completed."""
        if self._state is None:
            return False
        
        if self.task_type == "easy":
            return self._state.is_classified
        elif self.task_type == "medium":
            return self._state.is_prioritized
        elif self.task_type == "hard":
            return self._state.is_classified and self._state.is_prioritized
        else:
            return False
    
    def state(self) -> Optional[State]:
        """
        Get the current internal state (not exposed to agent).
        
        Returns:
            Current state or None if not initialized
        """
        return self._state
    
    def get_current_email(self) -> Optional[Dict[str, Any]]:
        """Get the current email data (for debugging)."""
        return self._current_email
    
    def close(self) -> None:
        """Close the environment (required by OpenEnv interface)."""
        # No resources to clean up in this simple environment
        pass
