---
title: DataCleaningEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DataCleaningEnv 🧹

**An OpenEnv environment where an AI agent cleans dirty tabular datasets across three real-world tasks.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

---

## Overview

`DataCleaningEnv` simulates the real-world task of data cleaning — one of the most time-consuming parts of any data pipeline. An agent receives a dirty dataset and must return a cleaned version, with partial credit for each category of fix it gets right.

**Why this matters:** Data cleaning is estimated to consume 60–80% of a data scientist's time. An agent that can reliably perform cleaning tasks has direct production value.

---

## Action & Observation Spaces

### Action (`DataCleaningAction`)
| Field | Type | Description |
|---|---|---|
| `cleaned_rows` | `List[Dict]` | The fully cleaned dataset rows |
| `operations_log` | `List[str]` | Description of each transformation applied |
| `dropped_indices` | `List[int]` | 0-based indices of rows removed |

### Observation (`DataCleaningObservation`)
| Field | Type | Description |
|---|---|---|
| `dirty_rows` | `List[Dict]` | The raw dirty dataset |
| `task_id` | `str` | Task identifier |
| `task_name` | `str` | Human-readable task name |
| `difficulty` | `str` | "easy" / "medium" / "hard" |
| `instructions` | `str` | Exact cleaning instructions |
| `schema_hint` | `Dict[str,str]` | Expected column types after cleaning |
| `feedback` | `str` | Grader feedback from previous step |
| `score` | `float` | Reward earned (0.0–1.0), -1 on reset |
| `done` | `bool` | Episode complete? |

---

## Tasks

### task_001 — Deduplicate & Standardise Phone Numbers (Easy)
**Dataset:** Customer table (7 rows, 2 exact duplicates)

**Cleaning required:**
- Remove exact duplicate rows (keep first occurrence)
- Normalise all phone numbers to `+1-XXX-XXX-XXXX` format

**Reward breakdown:**
- +0.40 — correct row count after deduplication
- +0.12 per phone correctly normalised (×5 rows = 0.60 max)

---

### task_002 — Fix Missing Values, Types & Inconsistent Strings (Medium)
**Dataset:** Employee salary table (6 rows)

**Cleaning required:**
- Fill `None` salary values with the median salary of non-null rows
- Strip `" years"` suffix from `age` column, cast to integer
- Normalise `department` to Title Case
- Convert mixed `"yes"`/`"no"`/`True`/`False` `active` column to boolean

**Reward breakdown:**
- +0.25 — all ages are integers
- +0.25 — all departments in Title Case
- +0.25 — missing salaries filled with correct median (92500.0)
- +0.25 — all `active` values are booleans

---

### task_003 — Multi-Issue: Outliers, Dates, Units & Integrity (Hard)
**Dataset:** Product orders table (8 rows, 4 distinct quality issues)

**Cleaning required:**
1. **Outliers** — Replace `price` values >10,000 or <0 with `null`
2. **Dates** — Normalise mixed date formats to ISO 8601 (`YYYY-MM-DD`)
3. **Units** — Convert `weight_kg` values >500 from grams to kg (÷1000)
4. **Integrity** — Drop rows where `customer_id` ∉ {101, 102, 103, 104}

**Reward breakdown:**
- +0.25 — referential integrity (invalid rows dropped)
- +0.25 — price outliers replaced with null
- +0.25 — all dates in ISO 8601
- +0.25 — all weights correctly in kg

---

## Reward Function

Each task scores 0.0–1.0 with **partial credit** — the agent earns points for each category of fix it gets right, even if it misses others. Maximum episode reward is **3.0**.

The reward function is fully deterministic and reproducible.

---

## Setup & Usage

### Local (no Docker)
```bash
cd data_cleaning_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
# Build
docker build -f server/Dockerfile -t data-cleaning-env .

# Run
docker run -p 7860:7860 data-cleaning-env
```

### API Endpoints
```
POST /reset   → Start new episode, receive first dirty dataset
POST /step    → Submit cleaned dataset, get reward + next task
GET  /state   → Current episode state
GET  /tasks   → Task metadata list
GET  /health  → Health check
```

### Run Baseline Inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your-api-key-here
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## Baseline Scores

| Task | Score |
|---|---|
| task_001 — Deduplicate & Phones (easy) | 1.00 |
| task_002 — Missing/Types/Strings (medium) | 0.88 |
| task_003 — Multi-Issue (hard) | 0.75 |
| **Total** | **2.63 / 3.00** |

---

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API base URL (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key / Hugging Face token |
| `ENV_BASE_URL` | Environment server (default: `http://localhost:7860`) |

---

## Project Structure

```
data_cleaning_env/
├── __init__.py
├── models.py                          # Pydantic Action/Observation/State
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml
├── inference.py                       # Baseline inference script
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                         # FastAPI application
    ├── data_cleaning_environment.py   # Core environment logic
    ├── tasks.py                       # Task bank + graders
    ├── requirements.txt
    └── Dockerfile
```
