"""Phase 10 — Trigger GitHub Actions workflow_dispatch from the dashboard.

PyGithub is imported lazily so that importing this module (e.g. during tests)
never requires the package to be installed unless a trigger is actually used.
"""

from __future__ import annotations

import time
from typing import Optional


def _secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch a secret from Streamlit, tolerant of running outside the app."""
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:  # noqa: BLE001
        import os
        return os.environ.get(key, default)


def _github_client():
    from github import Github  # lazy import
    token = _secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not configured in secrets.")
    return Github(token)


def trigger_workflow(
    workflow_filename: str,
    inputs: dict,
    ref: str = "main",
) -> str:
    """Trigger a workflow_dispatch event. Returns the run's html_url.

    Args:
        workflow_filename: e.g. "daily-pipeline.yml"
        inputs: dict of workflow_dispatch input values (all stringified)
        ref: git ref to run against (default "main")
    """
    repo_name = _secret("GITHUB_REPO")
    if not repo_name:
        raise RuntimeError("GITHUB_REPO not configured in secrets.")

    gh = _github_client()
    repo = gh.get_repo(repo_name)
    workflow = repo.get_workflow(workflow_filename)

    # Inputs must be strings for the GitHub API.
    str_inputs = {k: str(v) for k, v in (inputs or {}).items()}

    ok = workflow.create_dispatch(ref=ref, inputs=str_inputs)
    if not ok:
        raise RuntimeError("GitHub API rejected the workflow_dispatch request.")

    # Give GitHub a moment to register the run, then return the newest one.
    time.sleep(2)
    runs = workflow.get_runs()
    try:
        latest = runs[0]
        return latest.html_url
    except (IndexError, Exception):  # noqa: BLE001
        # Dispatch succeeded but the run isn't queryable yet — link to the tab.
        return f"https://github.com/{repo_name}/actions/workflows/{workflow_filename}"


def get_workflow_status(run_url: str) -> dict:
    """Poll a workflow run for status. Returns {status, conclusion, html_url}."""
    repo_name = _secret("GITHUB_REPO")
    if not repo_name:
        return {"status": "unknown", "conclusion": None, "html_url": run_url}

    # Extract the numeric run id from a URL like
    # https://github.com/<owner>/<repo>/actions/runs/<run_id>
    run_id = None
    if "/runs/" in run_url:
        tail = run_url.split("/runs/", 1)[1]
        digits = "".join(ch for ch in tail.split("/")[0] if ch.isdigit())
        run_id = int(digits) if digits else None

    if run_id is None:
        return {"status": "unknown", "conclusion": None, "html_url": run_url}

    try:
        gh = _github_client()
        repo = gh.get_repo(repo_name)
        run = repo.get_workflow_run(run_id)
        return {
            "status": run.status,            # queued | in_progress | completed
            "conclusion": run.conclusion,    # success | failure | None
            "html_url": run.html_url,
            "started_at": str(run.run_started_at) if run.run_started_at else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "conclusion": str(exc), "html_url": run_url}
