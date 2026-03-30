"""OpenEnv server entry point used by validators and deployment tooling."""

import uvicorn

from email_triage_env.server import app


def main() -> None:
    """Run the FastAPI app with uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
