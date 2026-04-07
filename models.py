"""
Data Cleaning Environment — Typed Models
Action, Observation, and State using Pydantic v2.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ──────────────────────────────────────────
# Action
# ──────────────────────────────────────────

class DataCleaningAction(BaseModel):
    """
    The agent submits a cleaned version of the dataset.

    Fields:
        cleaned_rows:    The cleaned dataset as a list of row dicts.
        operations_log:  List of human-readable descriptions of what was done,
                         e.g. ["Dropped row 3 (duplicate)", "Filled missing age with median"].
        dropped_indices: Indices of rows that were removed entirely.
    """
    cleaned_rows: List[Dict[str, Any]]
    operations_log: List[str]
    dropped_indices: Optional[List[int]] = None


# ──────────────────────────────────────────
# Observation
# ──────────────────────────────────────────

class DataCleaningObservation(BaseModel):
    """
    What the agent sees after each action (or on reset).

    Fields:
        dirty_rows:    The raw/dirty dataset as a list of row dicts.
        task_id:       Unique task identifier.
        task_name:     Human-readable task name.
        difficulty:    "easy" | "medium" | "hard"
        instructions:  What the agent should do to clean this dataset.
        schema_hint:   Expected column names and types after cleaning.
        feedback:      Grader feedback after a step (empty on reset).
        score:         Reward earned on this step (0.0–1.0), -1 on reset.
        done:          Whether the episode is complete.
    """
    dirty_rows: List[Dict[str, Any]]
    task_id: str
    task_name: str
    difficulty: str
    instructions: str
    schema_hint: Dict[str, str]      # {"column_name": "expected_type"}
    feedback: str = ""
    score: float = -1.0
    done: bool = False


# ──────────────────────────────────────────
# State
# ──────────────────────────────────────────

class DataCleaningState(BaseModel):
    """Internal episode state returned by state() endpoint."""
    episode_id: str = ""
    current_task_index: int = 0
    total_tasks: int = 3
    step_count: int = 0
    cumulative_reward: float = 0.0
    tasks_completed: List[str] = []
