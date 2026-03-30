# Email Triage OpenEnv Environment

An OpenEnv benchmark for a real-world productivity task: triaging incoming email. The environment evaluates whether an agent can read an email, classify it, assign the right priority, and handle multi-step decision making under ambiguity.

## Overview

This project is designed for OpenEnv-style agent evaluation with:

- a real-world task instead of a toy problem
- typed `Observation`, `Action`, `Reward`, and `State` models
- `step()`, `reset()`, and `state()` APIs
- three tasks with deterministic graders
- dense reward shaping for partial progress
- a reproducible baseline `inference.py`
- a FastAPI server and Dockerfile for deployment

The environment models a practical workflow humans actually do:

- inbox triage
- support queue routing
- personal assistant automation
- lightweight productivity tooling

## Environment Specification

### Objective

For each episode, the agent sees one email and must make structured decisions:

1. classify it as `spam`, `work`, or `personal`
2. assign priority `low`, `medium`, or `high`
3. optionally produce a reply on the hard task

Performance is measured two ways:

- dense step rewards during interaction
- final deterministic grader scores between `0.0` and `1.0`

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
- grader: `1.0` if classification is correct, else `0.0`

### Medium Task

Prioritization only.

- goal: choose the correct urgency level
- grader: `1.0` if priority is correct, else `0.0`

### Hard Task

Full triage.

- goal: classify the email, assign priority, and optionally reply
- grader: weighted score with classification worth `0.6` and priority worth `0.4`

## Reward Design

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

This gives a better learning signal for agent training while the final graders still produce simple deterministic scores in the `0.0` to `1.0` range.

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

If you already have the bundled virtual environment, you can use it directly.

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

### Example API calls

```bash
curl -X POST http://127.0.0.1:8000/reset ^
  -H "Content-Type: application/json" ^
  -d "{\"task\":\"easy\",\"seed\":42}"

curl -X POST http://127.0.0.1:8000/step ^
  -H "Content-Type: application/json" ^
  -d "{\"action\":{\"action_type\":\"classify\",\"label\":\"work\"}}"
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

From the repository root:

```bash
.venv\Scripts\openenv.exe validate
```

Additional checks:

```bash
.venv\Scripts\python.exe test_opnenv_validation.py
.venv\Scripts\python.exe demo_benchmark.py
```

## Docker

Build and run:

```bash
docker build -t email-triage-openenv .
docker run -p 8000:8000 email-triage-openenv
```

Note:

- Docker Desktop or another Docker daemon must be running for these commands to succeed.

## Hugging Face Spaces

This repo is structured for Docker-based deployment to Hugging Face Spaces.

Recommended setup:

- SDK: `Docker`
- hardware: CPU Basic is sufficient
- set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` in Space secrets if you want the LLM baseline enabled

## Why This Environment Fits The Problem Statement

- real-world utility: email triage is a practical daily workflow
- three tasks with deterministic graders: easy, medium, hard
- meaningful reward shaping: partial progress, sequencing, and undesirable-action penalties
- OpenEnv-compatible API and typed models
- baseline inference script included at repo root
- containerization included for deployment

## Extensibility

Possible next steps:

- more categories such as `finance`, `promotions`, or `social`
- multi-email threads
- sender metadata and domain trust signals
- stronger reply grading
- time-based urgency and SLA constraints

## License

MIT. See [LICENSE](/c:/Users/Admin/Hackathons/Scaler%20March%202026/LICENSE).
