"""
Inference script for the Email Triage OpenEnv environment.

Expected environment variables:
- API_BASE_URL: LLM endpoint, defaults to the Hugging Face router
- MODEL_NAME: model identifier to query
- HF_TOKEN: Hugging Face token or compatible API key

The script uses the OpenAI Python client for all LLM calls and falls back to a
deterministic heuristic baseline when credentials or network access are not
available.
"""

import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from email_triage_env.environment import EmailTriageEnv
from email_triage_env.models import Action, EmailCategory, Observation, PriorityLevel
from email_triage_env.tasks.graders import TaskEvaluator, EpisodeResult

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")

TEMPERATURE = 0.2
MAX_TOKENS = 200
RUNTIME_LIMIT_SECONDS = 20 * 60
DEBUG = True

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-api-key")


def observation_to_dict(observation: Observation | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize observations from either Pydantic models or plain dicts."""
    if isinstance(observation, Observation):
        return observation.model_dump()
    if isinstance(observation, Mapping):
        return dict(observation)
    raise TypeError(f"Unsupported observation type: {type(observation)!r}")


class BaselineAgent:
    """Simple baseline agent with optional LLM control and heuristic fallback."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(API_KEY)

        self.spam_keywords = ["win", "free", "prize", "congratulations", "click", "offer"]
        self.work_keywords = [
            "project",
            "deadline",
            "meeting",
            "team",
            "report",
            "client",
            "review",
            "hr",
            "benefits",
        ]
        self.high_priority_keywords = [
            "urgent",
            "critical",
            "important",
            "security",
            "alert",
            "deadline",
            "action required",
            "final attempt",
        ]
        self.medium_priority_keywords = [
            "meeting",
            "report",
            "review",
            "update",
            "reminder",
            "rescheduled",
            "renewal",
        ]
        self.low_priority_keywords = [
            "no rush",
            "when you're free",
            "whenever you're free",
            "optional",
            "social event",
            "bbq",
            "weekend",
            "casual friday",
        ]

    def act(self, observation: Observation | Mapping[str, Any], task_type: str = "easy") -> Action:
        payload = observation_to_dict(observation)
        if self.use_llm:
            return self._llm_act(payload, task_type)
        return self._heuristic_act(payload, task_type)

    def _heuristic_act(self, observation: dict[str, Any], task_type: str) -> Action:
        subject = observation["subject"].lower()
        body = observation["body"].lower()
        step_count = observation["step_count"]
        text = f"{subject} {body}"

        if task_type == "easy":
            return Action(action_type="classify", label=self._pick_category(text))

        if task_type == "medium":
            return Action(action_type="prioritize", level=self._pick_priority(text))

        if step_count == 0:
            return Action(action_type="classify", label=self._pick_category(text))
        if step_count == 1:
            return Action(action_type="prioritize", level=self._pick_priority(text))
        return Action(
            action_type="reply",
            text="Thanks for the email. I have reviewed it and will follow up if needed.",
        )

    def _llm_act(self, observation: dict[str, Any], task_type: str) -> Action:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self._system_prompt(task_type)},
                    {"role": "user", "content": self._build_prompt(observation)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            action_text = response.choices[0].message.content.strip()
            return self._parse_action(action_text, task_type)
        except Exception as exc:
            if DEBUG:
                print(f"[WARN] LLM call failed, using heuristics: {exc}")
            self.use_llm = False
            return self._heuristic_act(observation, task_type)

    def _pick_category(self, text: str) -> EmailCategory:
        if any(keyword in text for keyword in self.spam_keywords):
            return EmailCategory.SPAM
        if any(keyword in text for keyword in self.work_keywords):
            return EmailCategory.WORK
        return EmailCategory.PERSONAL

    def _pick_priority(self, text: str) -> PriorityLevel:
        if any(keyword in text for keyword in self.low_priority_keywords):
            return PriorityLevel.LOW
        if any(keyword in text for keyword in self.high_priority_keywords):
            return PriorityLevel.HIGH
        if any(keyword in text for keyword in self.medium_priority_keywords):
            return PriorityLevel.MEDIUM
        return PriorityLevel.LOW

    def _system_prompt(self, task_type: str) -> str:
        if task_type == "easy":
            return (
                "You classify emails as spam, work, or personal. "
                "Reply with exactly one action like classify(label=\"work\")."
            )
        if task_type == "medium":
            return (
                "You assign email priority as low, medium, or high. "
                "Reply with exactly one action like prioritize(level=\"high\")."
            )
        return (
            "You triage emails step by step. First classify, then prioritize, "
            "then optionally reply. Reply with exactly one action string."
        )

    def _build_prompt(self, observation: dict[str, Any]) -> str:
        prompt = (
            f"Email ID: {observation['email_id']}\n"
            f"Subject: {observation['subject']}\n"
            f"Body: {observation['body']}\n"
            f"Step: {observation['step_count'] + 1}\n"
        )
        history = observation.get("history") or []
        if history:
            prompt += "\nPrevious actions:\n"
            for action in history:
                prompt += f"- {action}\n"
        prompt += "\nWhat action should I take?"
        return prompt

    def _parse_action(self, action_text: str, task_type: str) -> Action:
        lowered = action_text.lower().strip()

        if "classify" in lowered:
            if "spam" in lowered:
                return Action(action_type="classify", label=EmailCategory.SPAM)
            if "work" in lowered:
                return Action(action_type="classify", label=EmailCategory.WORK)
            return Action(action_type="classify", label=EmailCategory.PERSONAL)

        if "prioritize" in lowered:
            if "high" in lowered:
                return Action(action_type="prioritize", level=PriorityLevel.HIGH)
            if "medium" in lowered:
                return Action(action_type="prioritize", level=PriorityLevel.MEDIUM)
            return Action(action_type="prioritize", level=PriorityLevel.LOW)

        if "reply" in lowered:
            start = action_text.find('"')
            end = action_text.rfind('"')
            reply_text = action_text[start + 1 : end] if start != -1 and end > start else "Acknowledged."
            return Action(action_type="reply", text=reply_text)

        if task_type == "medium":
            return Action(action_type="prioritize", level=PriorityLevel.MEDIUM)
        return Action(action_type="classify", label=EmailCategory.PERSONAL)


def run_baseline_evaluation() -> dict[str, Any]:
    """Run the baseline on all tasks and persist a JSON summary."""
    # Diagnostic output to stderr to keep stdout clean for validator
    import sys
    print("=" * 60, file=sys.stderr)
    print("Email Triage OpenEnv - Baseline Evaluation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    env = EmailTriageEnv()
    agent = BaselineAgent(use_llm=True)
    evaluator = TaskEvaluator()

    if not API_KEY:
        print("No API key provided. Using heuristic baseline only.", file=sys.stderr)
        agent.use_llm = False
    else:
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            print(f"[OK] LLM connected: {MODEL_NAME}", file=sys.stderr)
        except Exception as exc:
            print(f"[WARN] LLM connection failed: {exc}", file=sys.stderr)
            print("Falling back to heuristic baseline.", file=sys.stderr)
            agent.use_llm = False

    start_time = time.time()
    
    # We'll manually evaluate each task to output [START]/[STEP]/[END] blocks
    results = {}
    task_results = {}
    
    for task_type in ["easy", "medium", "hard"]:
        print(f"[START] task={task_type}", flush=True)
        
        # Reset environment with task type
        env.task_type = task_type
        observation = env.reset(seed={"easy": 42, "medium": 43, "hard": 44}[task_type])
        
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
            
            # Output step information
            print(f"[STEP] step={steps} reward={reward.total:.3f}", flush=True)
            
            # Check if actions were correct
            if info.get("is_classified"):
                if reward.classification and reward.classification > 0:
                    classification_correct = True
            
            if info.get("is_prioritized"):
                if reward.priority and reward.priority > 0:
                    priority_correct = True
        
        # Get score from grader
        grader = evaluator.graders[task_type]
        
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
        
        score = grader.grade(episode_result)
        
        print(f"[END] task={task_type} score={score:.1f} steps={steps}", flush=True)
        
        task_results[task_type] = {
            "task_type": task_type,
            "score": score,
            "total_reward": total_reward,
            "steps_taken": steps,
            "max_steps": env.max_steps,
            "classification_correct": classification_correct,
            "priority_correct": priority_correct,
            "episode_result": episode_result
        }
    
    # Calculate average score
    scores = [task_results[task]["score"] for task in task_results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    results = {
        "task_results": task_results,
        "average_score": avg_score,
        "timestamp": time.time()
    }
    
    elapsed_time = time.time() - start_time

    # Diagnostic output to stderr
    import sys
    print("\n" + "=" * 60, file=sys.stderr)
    print("EVALUATION RESULTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for task_type in ["easy", "medium", "hard"]:
        task_result = results["task_results"][task_type]
        print(f"\n{task_type.upper()} TASK:", file=sys.stderr)
        print(f"  Score: {task_result['score']:.3f}", file=sys.stderr)
        print(f"  Total Reward: {task_result['total_reward']:.3f}", file=sys.stderr)
        print(f"  Steps: {task_result['steps_taken']}/{task_result['max_steps']}", file=sys.stderr)
        print(f"  Classification Correct: {task_result['classification_correct']}", file=sys.stderr)
        print(f"  Priority Correct: {task_result['priority_correct']}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print(f"Average Score: {results['average_score']:.3f}", file=sys.stderr)
    print(f"Total Time: {elapsed_time:.2f} seconds", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    with open("baseline_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    print("\nResults saved to: baseline_results.json", file=sys.stderr)
    if elapsed_time > RUNTIME_LIMIT_SECONDS:
        print(f"[WARN] Runtime ({elapsed_time:.2f}s) exceeds 20 minute limit.", file=sys.stderr)
    else:
        print(f"[OK] Runtime ({elapsed_time:.2f}s) within 20 minute limit.", file=sys.stderr)

    return results


if __name__ == "__main__":
    import sys
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable not set!", file=sys.stderr)
        sys.exit(1)

    print(f"API Base URL: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    run_baseline_evaluation()
