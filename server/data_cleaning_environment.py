"""
Data Cleaning Environment — Dynamic Core
=========================================
Upgraded environment that:
1. Generates a FRESH dataset on every reset() using the dynamic generator
2. Tracks agent performance across episodes for curriculum difficulty
3. Falls back cleanly to static datasets if LLM is unavailable
4. Remains fully OpenEnv-spec compliant
"""

import sys, os, uuid, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from server.dataset_generator import generate_dynamic_task
from server.dynamic_grader import grade_dynamic_task

DIFFICULTY_SEQUENCE = ["easy", "medium", "hard"]


class DataCleaningEnvironment:
    def __init__(self):
        self._state = DataCleaningState()
        self._tasks = []
        self._episode_seed = None
        self._episode_history = []

    def reset(self) -> DataCleaningObservation:
        episode_seed = random.randint(0, 999999)
        self._episode_seed = episode_seed
        self._tasks = []
        for i, difficulty in enumerate(DIFFICULTY_SEQUENCE):
            task = generate_dynamic_task(difficulty=difficulty, seed=episode_seed + i)
            self._tasks.append(task)
        self._state = DataCleaningState(
            episode_id=str(uuid.uuid4()),
            current_task_index=0,
            total_tasks=len(self._tasks),
            step_count=0,
            cumulative_reward=0.0,
            tasks_completed=[],
        )
        return self._obs_for_task(0)

    def step(self, action: DataCleaningAction) -> Tuple[DataCleaningObservation, float, bool, dict]:
        idx = self._state.current_task_index
        if idx >= len(self._tasks):
            return (
                DataCleaningObservation(
                    dirty_rows=[], task_id="done", task_name="Episode complete",
                    difficulty="", instructions="", schema_hint={},
                    feedback="Episode already complete.", score=0.0, done=True,
                ),
                0.0, True, {"warning": "step called after episode end"},
            )
        task = self._tasks[idx]
        reward, feedback = grade_dynamic_task(task, action.model_dump())
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.tasks_completed.append(task["task_id"])
        self._state.current_task_index += 1
        done = self._state.current_task_index >= len(self._tasks)
        if done:
            self._episode_history.append({
                "episode_id": self._state.episode_id,
                "total_reward": round(self._state.cumulative_reward, 3),
                "episode_seed": self._episode_seed,
            })
        if not done:
            next_obs = self._obs_for_task(self._state.current_task_index)
            next_obs.feedback = f"[Previous task feedback]\n{feedback}"
            next_obs.score = reward
        else:
            next_obs = DataCleaningObservation(
                dirty_rows=[], task_id="episode_end",
                task_name="Episode Complete", difficulty="",
                instructions="", schema_hint={},
                feedback=(
                    f"[Final task feedback]\n{feedback}\n\n"
                    f"Episode complete! Total reward: {self._state.cumulative_reward:.2f}/{len(self._tasks):.1f}\n"
                    f"Episode seed: {self._episode_seed}"
                ),
                score=reward, done=True,
            )
        info = {
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "difficulty": task["difficulty"],
            "domain": task.get("_domain", "unknown"),
            "dynamic": task.get("_dynamic", False),
            "step_count": self._state.step_count,
            "cumulative_reward": self._state.cumulative_reward,
            "episode_seed": self._episode_seed,
        }
        return next_obs, reward, done, info

    def state(self) -> DataCleaningState:
        return self._state

    def curriculum_stats(self) -> dict:
        if not self._episode_history:
            return {"episodes": 0, "avg_reward": 0.0, "best_reward": 0.0, "history": []}
        rewards = [e["total_reward"] for e in self._episode_history]
        return {
            "episodes": len(rewards),
            "avg_reward": round(sum(rewards) / len(rewards), 3),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "history": self._episode_history[-20:],
        }

    def _obs_for_task(self, idx: int) -> DataCleaningObservation:
        import copy
        t = self._tasks[idx]
        return DataCleaningObservation(
            dirty_rows=copy.deepcopy(t["dirty_rows"]),
            task_id=t["task_id"],
            task_name=t["task_name"],
            difficulty=t["difficulty"],
            instructions=t["instructions"],
            schema_hint=t["schema_hint"],
        )
