"""
inference.py — Baseline Inference Script for DataCleaningEnv

Uses OpenAI client to run a model against all 3 cleaning tasks.
Emits structured stdout logs in [START] / [STEP] / [END] format.

Required environment variables:
  API_BASE_URL  — LLM API endpoint (OpenAI-compatible)
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key / Hugging Face token

Optional:
  ENV_BASE_URL  — Environment server URL (default: http://localhost:7860)

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=sk-...
  python inference.py
"""

import os, json, time, requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL",  "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset():
    r = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: dict):
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM prompts ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data engineer specialising in data cleaning.

You will receive a dirty dataset (list of JSON row objects) along with cleaning instructions.
You MUST respond with a JSON object only — absolutely no prose, no markdown code fences.

Your response schema:
{
  "cleaned_rows": [ { ...row dict... }, ... ],
  "operations_log": ["<description of operation 1>", "<description of operation 2>", ...],
  "dropped_indices": [<0-based index of dropped rows>, ...]
}

Rules:
- Apply EVERY instruction listed precisely.
- In cleaned_rows, preserve all column names exactly as given.
- In operations_log, describe each transformation you performed.
- In dropped_indices, list the 0-based index of any rows you removed.
- Output ONLY the JSON object. No other text whatsoever.
"""

def build_user_message(obs: dict) -> str:
    return (
        f"Task: {obs['task_name']} (Difficulty: {obs['difficulty']})\n\n"
        f"Instructions:\n{obs['instructions']}\n\n"
        f"Expected schema after cleaning:\n{json.dumps(obs['schema_hint'], indent=2)}\n\n"
        f"Dirty dataset ({len(obs['dirty_rows'])} rows):\n"
        f"{json.dumps(obs['dirty_rows'], indent=2)}\n\n"
        f"Return your cleaned JSON now."
    )

def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()

def parse_action(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
    return json.loads(clean)

# ── Main episode loop ─────────────────────────────────────────────────────────

def run_episode():
    episode_start = time.time()
    total_reward  = 0.0
    step_num      = 0

    # ── [START] ──────────────────────────────────────────────────────────────
    print("[START] " + json.dumps({
        "task":      MODEL_NAME,
        "model":     MODEL_NAME,
        "env":       "DataCleaningEnv",
        "timestamp": episode_start,
    }), flush=True)

    obs      = env_reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs.get("done", False):
        step_num += 1
        step_start = time.time()

        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        raw_response = ""
        action = {"cleaned_rows": [], "operations_log": [], "dropped_indices": []}
        try:
            raw_response = call_llm(messages)
            action       = parse_action(raw_response)
        except Exception as e:
            print("[STEP] " + json.dumps({
                "step":    step_num,
                "task_id": obs.get("task_id"),
                "error":   str(e),
                "reward":  0.0,
            }), flush=True)

        messages.append({"role": "assistant", "content": raw_response})

        result    = env_step(action)
        reward    = result["reward"]
        done      = result["done"]
        info      = result["info"]
        next_obs  = result["observation"]
        total_reward += reward

        # ── [STEP] ───────────────────────────────────────────────────────────
        print("[STEP] " + json.dumps({
            "step":              step_num,
            "task_id":           info.get("task_id"),
            "difficulty":        info.get("difficulty"),
            "reward":            reward,
            "cumulative_reward": total_reward,
            "done":              done,
            "duration_s":        round(time.time() - step_start, 2),
            "action_summary": {
                "rows_returned":   len(action.get("cleaned_rows", [])),
                "rows_dropped":    len(action.get("dropped_indices") or []),
                "operations_logged": len(action.get("operations_log", [])),
            },
        }), flush=True)

        obs = next_obs

        # Feed grader feedback back into context
        if obs.get("feedback"):
            messages.append({
                "role":    "user",
                "content": f"[Grader feedback from previous task]\n{obs['feedback']}",
            })

    # ── [END] ─────────────────────────────────────────────────────────────────
    print("[END] " + json.dumps({
        "task":              MODEL_NAME,
        "total_steps":       step_num,
        "total_reward":      round(total_reward, 3),
        "max_possible":      3.0,
        "score":             round(total_reward / 3.0, 3),
        "duration_s":        round(time.time() - episode_start, 2),
    }), flush=True)

    return total_reward


if __name__ == "__main__":
    score = run_episode()
