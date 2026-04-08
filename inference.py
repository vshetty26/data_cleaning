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

TASK_NAME  = "data-cleaning"
BENCHMARK  = "DataCleaningEnv"

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

def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — validator rejects exact 0.0 and 1.0."""
    s = round(score, 4)
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    return s

# ── Main episode loop ─────────────────────────────────────────────────────────

def run_episode():
    step_num     = 0
    total_reward = 0.0
    step_rewards = []
    last_error   = "null"

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    obs      = env_reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while not obs.get("done", False):
            step_num  += 1
            last_error = "null"

            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            raw_response = ""
            action = {"cleaned_rows": [], "operations_log": [], "dropped_indices": []}
            try:
                raw_response = call_llm(messages)
                action       = parse_action(raw_response)
            except Exception as e:
                last_error = str(e).replace("\n", " ")

            messages.append({"role": "assistant", "content": raw_response})

            result   = env_step(action)
            reward   = result["reward"]
            done     = result["done"]
            next_obs = result["observation"]

            total_reward += reward
            step_rewards.append(reward)

            action_label = obs.get("task_id", f"step{step_num}")
            done_str     = "true" if done else "false"

            # ── [STEP] ───────────────────────────────────────────────────────
            print(
                f"[STEP] step={step_num} action={action_label} "
                f"reward={reward:.2f} done={done_str} error={last_error}",
                flush=True,
            )

            obs = next_obs

            # Feed grader feedback back into context
            if obs.get("feedback"):
                messages.append({
                    "role":    "user",
                    "content": f"[Grader feedback from previous task]\n{obs['feedback']}",
                })

    except Exception as e:
        last_error = str(e).replace("\n", " ")

    # ── [END] ─────────────────────────────────────────────────────────────────
    final_score  = _clamp(total_reward / 3.0)
    rewards_str  = ",".join(f"{r:.2f}" for r in step_rewards)
    success_str  = "true" if total_reward > 0 else "false"

    print(
        f"[END] success={success_str} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return total_reward


if __name__ == "__main__":
    run_episode()
