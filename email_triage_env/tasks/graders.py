"""
Deterministic graders for Email Triage tasks.

Each grader produces a score between 0.0 and 1.0 based on agent performance.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..models import EmailCategory, PriorityLevel


@dataclass
class EpisodeResult:
    """Result of a complete episode."""
    task_type: str
    total_reward: float
    steps_taken: int
    max_steps: int
    is_classified: bool
    is_prioritized: bool
    is_replied: Optional[bool] = None
    classification_correct: Optional[bool] = None
    priority_correct: Optional[bool] = None
    action_history: List[Dict[str, Any]] = None


class BaseGrader:
    """Base class for task graders."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
    
    def grade(self, episode_result: EpisodeResult) -> float:
        """
        Grade an episode and return score between 0.0 and 1.0.
        
        Args:
            episode_result: Result of the completed episode
            
        Returns:
            Score between 0.0 and 1.0
        """
        raise NotImplementedError


class EasyTaskGrader(BaseGrader):
    """Grader for Easy Task (Classification only)."""
    
    def __init__(self):
        super().__init__("easy")
    
    def grade(self, episode_result: EpisodeResult) -> float:
        """
        Score = 1.0 if classification is correct, else 0.0
        Clamped to be strictly between 0 and 1 for validator requirements.
        
        Args:
            episode_result: Must contain classification_correct field
            
        Returns:
            0.999 if correct classification, 0.001 otherwise
        """
        if episode_result.classification_correct:
            return 0.999
        return 0.001


class MediumTaskGrader(BaseGrader):
    """Grader for Medium Task (Prioritization only)."""
    
    def __init__(self):
        super().__init__("medium")
    
    def grade(self, episode_result: EpisodeResult) -> float:
        """
        Score = 1.0 if priority is correct, else 0.0
        Clamped to be strictly between 0 and 1 for validator requirements.
        
        Args:
            episode_result: Must contain priority_correct field
            
        Returns:
            0.999 if correct priority, 0.001 otherwise
        """
        if episode_result.priority_correct:
            return 0.999
        return 0.001


class HardTaskGrader(BaseGrader):
    """Grader for Hard Task (Full triage)."""
    
    def __init__(self):
        super().__init__("hard")
    
    def grade(self, episode_result: EpisodeResult) -> float:
        """
        Weighted scoring:
        - Classification: 0.6 weight
        - Priority: 0.4 weight
        
        Final score = (0.6 × classification correctness) + (0.4 × priority correctness)
        Clamped to be strictly between 0 and 1 for validator requirements.
        
        Args:
            episode_result: Must contain classification_correct and priority_correct fields
            
        Returns:
            Weighted score strictly between 0.0 and 1.0
        """
        classification_score = 1.0 if episode_result.classification_correct else 0.0
        priority_score = 1.0 if episode_result.priority_correct else 0.0
        
        weighted_score = (0.6 * classification_score) + (0.4 * priority_score)
        
        # Clamp to strictly between 0 and 1 for validator requirements
        if weighted_score >= 1.0:
            return 0.999
        elif weighted_score <= 0.0:
            return 0.001
        else:
            return weighted_score


class TaskEvaluator:
    """
    Evaluator for running agents through tasks and computing scores.
    
    This class simulates episodes and uses graders to produce deterministic scores.
    """
    
    def __init__(self):
        self.graders = {
            "easy": EasyTaskGrader(),
            "medium": MediumTaskGrader(),
            "hard": HardTaskGrader()
        }
    
    def evaluate_episode(self, env, agent, task_type: str = "easy", seed: int = 42) -> Dict[str, Any]:
        """
        Run a single episode with the given agent and environment.
        
        Args:
            env: EmailTriageEnv instance
            agent: Agent that can take actions
            task_type: "easy", "medium", or "hard"
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with episode results and score
        """
        # Reset environment with task type
        env.task_type = task_type
        observation = env.reset(seed=seed)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        # Track correctness
        classification_correct = False
        priority_correct = False
        
        while not done and steps < env.max_steps:
            # Get action from agent
            action = agent.act(observation, task_type)
            
            # Execute action
            observation, reward, done, info = env.step(action)
            
            # Update tracking
            total_reward += reward.total
            steps += 1
            
            # Check if actions were correct
            if info.get("is_classified"):
                # Check if classification was correct
                # This would require comparing agent's classification with ground truth
                # For simplicity, we assume agent gets it right if reward.classification > 0
                if reward.classification and reward.classification > 0:
                    classification_correct = True
            
            if info.get("is_prioritized"):
                if reward.priority and reward.priority > 0:
                    priority_correct = True
        
        # Create episode result
        state = env.state()

        episode_result = EpisodeResult(
            task_type=task_type,
            total_reward=total_reward,
            steps_taken=steps,
            max_steps=env.max_steps,
            is_classified=state.is_classified if state else False,
            is_prioritized=state.is_prioritized if state else False,
            is_replied=state.is_replied if state and task_type == "hard" else None,
            classification_correct=classification_correct,
            priority_correct=priority_correct,
            action_history=state.action_history if state else []
        )
        
        # Get score from grader
        grader = self.graders[task_type]
        score = grader.grade(episode_result)
        
        return {
            "task_type": task_type,
            "score": score,
            "total_reward": total_reward,
            "steps_taken": steps,
            "max_steps": env.max_steps,
            "classification_correct": classification_correct,
            "priority_correct": priority_correct,
            "episode_result": episode_result
        }
    
    def evaluate_all_tasks(self, env, agent, seeds: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Evaluate agent on all three tasks.
        
        Args:
            env: EmailTriageEnv instance
            agent: Agent that can take actions
            seeds: Optional dict of seeds for each task
            
        Returns:
            Dictionary with scores for all tasks
        """
        if seeds is None:
            seeds = {"easy": 42, "medium": 43, "hard": 44}
        
        results = {}
        for task_type in ["easy", "medium", "hard"]:
            results[task_type] = self.evaluate_episode(
                env, agent, task_type, seeds[task_type]
            )
        
        # Calculate average score
        scores = [results[task]["score"] for task in results]
        avg_score = sum(scores) / len(scores)
        
        return {
            "task_results": results,
            "average_score": avg_score,
            "scores": {
                "easy": results["easy"]["score"],
                "medium": results["medium"]["score"],
                "hard": results["hard"]["score"]
            }
        }
