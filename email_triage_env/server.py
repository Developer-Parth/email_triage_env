"""FastAPI server for the Email Triage OpenEnv environment."""

from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import EmailTriageEnv
from .models import Action, Observation, Reward


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""
    seed: int | None = None
    task: str = "easy"
    max_steps: int = 8


class StepRequest(BaseModel):
    """Request model for step endpoint."""
    action: Dict[str, Any]


class OpenEnvServer:
    """OpenEnv HTTP server for Email Triage environment."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Email Triage OpenEnv Environment",
            description="Real-world email triage simulation for AI agents",
            version="0.1.0"
        )
        self.env = None
        self.current_task = "easy"
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup OpenEnv API routes."""
        
        @self.app.post("/reset")
        async def reset(reset_request: ResetRequest | None = None):
            """Reset the environment and start a new episode."""
            try:
                reset_request = reset_request or ResetRequest()
                self.current_task = reset_request.task
                self.env = EmailTriageEnv(
                    task_type=reset_request.task,
                    max_steps=reset_request.max_steps
                )
                
                observation = self.env.reset(seed=reset_request.seed)
                
                return {
                    "observation": observation.model_dump(),
                    "info": {
                        "task_type": reset_request.task,
                        "max_steps": reset_request.max_steps
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/step")
        async def step(step_request: StepRequest):
            """Execute an action and return the result."""
            if self.env is None:
                raise HTTPException(
                    status_code=400,
                    detail="Environment not initialized. Call /reset first."
                )
            
            try:
                # Parse action from request
                action_dict = step_request.action
                action = Action(**action_dict)
                
                # Execute step
                observation, reward, done, info = self.env.step(action)
                
                return {
                    "observation": observation.model_dump(),
                    "reward": reward.model_dump(),
                    "done": done,
                    "info": info
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/state")
        async def get_state():
            """Get the current internal state (for debugging)."""
            if self.env is None:
                raise HTTPException(
                    status_code=400,
                    detail="Environment not initialized. Call /reset first."
                )
            
            state = self.env.state()
            if state:
                return {"state": state.model_dump()}
            else:
                return {"state": None}
        
        @self.app.get("/spec")
        async def get_spec():
            """Get environment specification."""
            return {
                "name": "email-triage",
                "version": "0.1.0",
                "description": "Email triage simulation for AI agents",
                "action_space": {
                    "classify": {
                        "label": ["spam", "work", "personal"]
                    },
                    "prioritize": {
                        "level": ["low", "medium", "high"]
                    },
                    "reply": {
                        "text": "string"
                    }
                },
                "observation_space": {
                    "email_id": "string",
                    "subject": "string",
                    "body": "string",
                    "step_count": "int",
                    "history": "optional list of previous actions"
                },
                "reward_range": [-0.3, 0.6],
                "max_steps": 8,
                "tasks": ["easy", "medium", "hard"]
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy"}
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Email Triage OpenEnv Environment",
                "endpoints": [
                    "POST /reset - Reset environment",
                    "POST /step - Execute action",
                    "GET /state - Get internal state",
                    "GET /spec - Get environment spec",
                    "GET /health - Health check"
                ]
            }


# Create the FastAPI app instance
app = OpenEnvServer().app


def main() -> None:
    """Run the OpenEnv server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
