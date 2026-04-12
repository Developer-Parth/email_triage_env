---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: OpenEnv benchmark for multi-step email triage
---

# Email Triage OpenEnv Benchmark

## 🚀 Scaler March 2026 Hackathon Submission

This project was built as part of the **Scaler March 2026 Hackathon**.

**Author:** Parth Thukral  
**Type:** OpenEnv Benchmark Environment  
**Focus:** Evaluating LLM agents on multi-step decision-making tasks

---

## ⚡ TL;DR

A deterministic benchmark that tests whether LLM agents can make consistent, multi-step decisions under ambiguity, not just classify emails correctly.

- enforces action sequencing and dependency constraints
- penalizes redundant or inconsistent behavior
- rewards efficient, correct decision-making
- designed to expose shallow or exploitative strategies

---

This benchmark is designed to test decision quality, not just accuracy.

> Designed to evaluate multi-step reasoning and decision consistency in LLM agents under real-world ambiguity.
>
> Suitable for benchmarking agent performance, reward shaping strategies, and decision consistency under constrained interaction loops.

This environment models a real-world productivity task: triaging incoming email. It evaluates whether an agent can read content, classify it, assign priority, and handle multi-step decisions under ambiguity.

## Overview

This is an OpenEnv-compatible benchmark for evaluating LLM agents on real-world, multi-step decision tasks.

**Key characteristics:**
- real-world task (email triage)
- structured Observation / Action / Reward / State models
- step-based interaction loop (`reset → step → done`)
- deterministic grading across three difficulty levels
- dense reward shaping for partial progress
- fully deployable via FastAPI + Docker


## What Makes This Different

Most benchmarks reward correctness.

This environment evaluates behavior.

It enforces:
- step-by-step reasoning instead of one-shot answers
- penalties for redundant or repeated actions
- consistency constraints across decisions
- efficiency through step costs
- memory of past mistakes affecting future reward

As a result, naive strategies (random guessing, repetition, late correction) fail to achieve high scores.

## Environment Specification

### Objective

For each episode, the agent sees one email and must make structured decisions:

1. classify it as `spam`, `work`, or `personal`
2. assign priority `low`, `medium`, or `high`
3. optionally produce a reply on the hard task

Performance is measured two ways:

- dense step rewards during interaction
- final deterministic grader scores strictly between `0.0` and `1.0`

### State

The internal environment state tracks:

- current email ID, subject, and body
- ground-truth category and priority
- completion flags: `is_classified`, `is_prioritized`, `is_replied`
- action history
- step count and step limit
- current classification and priority selections for consistency checks

The full state is available through the OpenEnv `state()` API for debugging and validation, but the agent does not directly observe the ground truth during normal play.

### Observation Space

The agent receives:

- `email_id`
- `subject`
- `body`
- `step_count`
- `history` of recent actions, when available

### Action Space

The environment accepts structured actions:

- `classify(label)` where `label` is one of `spam`, `work`, `personal`
- `prioritize(level)` where `level` is one of `low`, `medium`, `high`
- `reply(text)` for the hard task

Invalid or malformed actions are penalized.

### Episode Flow

1. `reset()` samples an email from the selected dataset split
2. the agent receives an observation
3. the agent acts through `step(action)`
4. the environment returns `(observation, reward, done, info)`
5. the episode ends when the task is complete or the maximum step limit is reached

## Tasks

### Easy Task

Classification only.

- goal: choose the correct email category
- grader: `0.999` if classification is correct, else `0.001`

### Medium Task

Prioritization only.

- goal: choose the correct urgency level
- grader: `0.999` if priority is correct, else `0.001`

### Hard Task

Full triage.

- goal: classify the email, assign priority, and optionally reply
- grader: weighted score with classification worth `0.6` and priority worth `0.4`

## Reward Design

**Summary:** Correct behavior yields positive reward (~1.0), random strategies are penalized (negative reward), ensuring meaningful learning signals.

The benchmark uses dense, shaped rewards so agents receive signal across the full trajectory instead of only at episode end.

Core components:

- classification reward
- priority reward
- sequence reward
- invalid action penalty
- completion bonus

Benchmark extensions:

- step cost to encourage efficiency
- consistency penalties for erratic behavior
- difficulty-scaled rewards to make ambiguous emails more informative

This gives a better learning signal for agent training while the final graders still produce simple deterministic scores strictly inside the `(0, 1)` range.

## Dataset

The built-in dataset contains 20 emails:

- 10 train emails
- 10 test emails

It includes:

- straightforward spam/work/personal examples
- ambiguous cases
- difficult mixed-signal emails
- edge cases intended to test generalization

Dataset modes:

- `train`
- `test`
- `mixed`

## Project Structure

```text
.
├── email_triage_env/
│   ├── __init__.py
│   ├── environment.py
│   ├── models.py
│   ├── requirements.txt
│   ├── server.py
│   └── tasks/
│       ├── __init__.py
│       └── graders.py
├── server/
│   ├── __init__.py
│   └── app.py
├── demo_benchmark.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Local Usage

### Run the baseline environment scripts

```bash
.venv\Scripts\python.exe test_environment.py
.venv\Scripts\python.exe test_final_enhancements.py
.venv\Scripts\python.exe test_reward_balance.py
.venv\Scripts\python.exe test_exploit_resistance.py
```

### Start the OpenEnv server

```bash
.venv\Scripts\python.exe -m uvicorn email_triage_env.server:app --host 0.0.0.0 --port 8000
```

or:

```bash
.venv\Scripts\python.exe -m server.app
```

## Baseline Inference

The baseline script uses the OpenAI Python client and reads configuration from environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
set HF_TOKEN=your-token
.venv\Scripts\python.exe inference.py
```

Behavior:

- tries an LLM-backed agent first
- falls back to deterministic heuristics when credentials or network access are unavailable
- writes `baseline_results.json`
- uses fixed seeds for reproducible task scores

## Validation

```bash
.venv\Scripts\openenv.exe validate
.venv\Scripts\python.exe test_opnenv_validation.py
.venv\Scripts\python.exe demo_benchmark.py
```

## 🔌 API Usage

All endpoints are OpenEnv-compatible and return structured JSON responses.

### Health Check
GET /health

### Reset Environment
POST /reset  
Optional body:
```json
{"task": "easy", "seed": 42}
```

### Take Step

POST /step
Body:

```json
{
  "action": {
    "action_type": "classify",
    "label": "personal"
  }
}
```

### Get State

GET /state

## Docker

Build and run:

```bash
docker build -t email-triage-openenv .
docker run -p 8000:8000 email-triage-openenv
```

## Hugging Face Spaces

This repo is structured for Docker-based deployment to Hugging Face Spaces.

Recommended setup:

- SDK: `Docker`
- hardware: CPU Basic is sufficient
- set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` in Space secrets if you want the LLM baseline enabled

## 🏁 Submission Status

This environment:

- passes OpenEnv validation
- successfully deploys via Docker on Hugging Face Spaces
- supports full agent interaction through API endpoints
- was tested end-to-end including inference and grading pipeline

Built, debugged, and deployed under hackathon constraints.

---

## 🔗 Links

- GitHub Repository: https://github.com/Developer-Parth/email_triage_env
- Hugging Face Space: https://huggingface.co/spaces/Developer-Parth/email_triage_env

## Why This Stands Out

Unlike typical benchmarks, this environment:

- evaluates sequential decision-making, not just single-step accuracy
- penalizes redundant or inefficient actions
- enforces consistency across multiple steps
- introduces memory of past mistakes into future rewards
- exposes shallow strategies like guessing, repetition, or late correction

As a result, agents must behave intelligently across the full interaction, not just produce correct final answers.

## Extensibility

Possible next steps:

- more categories such as `finance`, `promotions`, or `social`
- multi-email threads
- sender metadata and domain trust signals
- stronger reply grading
- time-based urgency and SLA constraints

## License

[MIT](https://github.com/Developer-Parth/email_triage_env/blob/main/LICENSE)
