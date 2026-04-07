"""
Data Cleaning Environment — FastAPI Server

Endpoints:
  POST /reset         → DataCleaningObservation  (fresh dataset generated)
  POST /step          → {observation, reward, done, info}
  GET  /state         → DataCleaningState
  GET  /tasks         → current episode task metadata
  GET  /curriculum    → agent performance history across episodes
  GET  /health        → {"status": "ok"}
  GET  /              → Interactive web UI (judges can play with it live)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from server.data_cleaning_environment import DataCleaningEnvironment

app = FastAPI(
    title="DataCleaningEnv",
    description=(
        "OpenEnv — Dynamic Data Cleaning RL Environment. "
        "Generates a fresh dataset on every reset(). "
        "3 tasks, easy→medium→hard, partial-progress rewards."
    ),
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = DataCleaningEnvironment()


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "dynamic_generation": True}


@app.post("/reset", response_model=DataCleaningObservation)
def reset():
    """Start a new episode. A brand-new dirty dataset is generated for each difficulty."""
    return env.reset()


@app.post("/step")
def step(action: DataCleaningAction):
    """Submit a cleaned dataset for the current task."""
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state", response_model=DataCleaningState)
def state():
    return env.state()


@app.get("/curriculum")
def curriculum():
    """Agent performance history — useful for tracking RL training progress."""
    return env.curriculum_stats()


@app.get("/tasks")
def list_tasks():
    """Current episode task metadata (after reset)."""
    return [
        {
            "task_id":      t["task_id"],
            "task_name":    t["task_name"],
            "difficulty":   t["difficulty"],
            "instructions": t["instructions"],
            "schema_hint":  t["schema_hint"],
            "dynamic":      t.get("_dynamic", False),
            "domain":       t.get("_domain", "static"),
        }
        for t in (env._tasks or [])
    ]


# ── Interactive Web UI ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def ui():
    """Live interactive playground for judges — no code required."""
    return HTMLResponse(content=_UI_HTML, status_code=200)


_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DataCleaningEnv — Interactive Playground</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #1a6fbf; --accent2: #238636; --text: #e6edf3;
    --muted: #8b949e; --easy: #3fb950; --medium: #d29922; --hard: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 700; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 600; }
  .badge-blue { background: #1a3a5c; color: #79c0ff; }
  .badge-green { background: #1a3a25; color: #3fb950; }
  main { max-width: 1100px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .card-header { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
  .card-header h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .card-body { padding: 16px; }
  .pill { font-size: 11px; font-weight: 700; padding: 2px 10px; border-radius: 20px; }
  .pill-easy   { background: #1a3a1a; color: var(--easy); }
  .pill-medium { background: #3a2a0a; color: var(--medium); }
  .pill-hard   { background: #3a1212; color: var(--hard); }
  pre { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 12px; overflow: auto; max-height: 240px; color: #79c0ff; white-space: pre-wrap; word-break: break-word; }
  textarea { width: 100%; background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: 'Courier New', monospace; font-size: 12px; color: #79c0ff; resize: vertical; min-height: 200px; }
  .btn { padding: 8px 18px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; border: none; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-success { background: var(--accent2); color: white; }
  .btn-row { display: flex; gap: 10px; margin-top: 12px; }
  .reward-bar-wrap { background: #0d1117; border-radius: 4px; height: 8px; margin: 8px 0; overflow: hidden; }
  .reward-bar { height: 100%; background: linear-gradient(90deg, #1a6fbf, #3fb950); border-radius: 4px; transition: width 0.4s; }
  .stat-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin: 4px 0; }
  .stat-val { color: var(--text); font-weight: 600; }
  .log { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 11px; font-family: monospace; max-height: 160px; overflow-y: auto; color: var(--muted); }
  .log .ok   { color: #3fb950; }
  .log .err  { color: #f85149; }
  .log .info { color: #79c0ff; }
  .full-width { grid-column: 1 / -1; }
  .task-meta { font-size: 12px; color: var(--muted); line-height: 1.6; margin-top: 8px; }
  .task-meta strong { color: var(--text); }
  .sep { border: none; border-top: 1px solid var(--border); margin: 10px 0; }
  .hint-table { width: 100%; font-size: 11px; border-collapse: collapse; margin-top: 8px; }
  .hint-table td { padding: 3px 8px; border: 1px solid var(--border); }
  .hint-table td:first-child { color: var(--muted); width: 40%; }
</style>
</head>
<body>
<header>
  <h1>DataCleaningEnv</h1>
  <span class="badge badge-blue">OpenEnv v1</span>
  <span class="badge badge-green">Dynamic Generation</span>
  <span style="margin-left:auto;font-size:12px;color:var(--muted)">Interactive Playground</span>
</header>
<main>

  <!-- Left col: Current Task -->
  <div class="card">
    <div class="card-header">
      <h2>Current Task</h2>
      <span id="diff-pill" class="pill">—</span>
    </div>
    <div class="card-body">
      <div class="stat-row"><span>Task</span><span class="stat-val" id="task-name">—</span></div>
      <div class="stat-row"><span>Task ID</span><span class="stat-val" id="task-id">—</span></div>
      <div class="stat-row"><span>Domain</span><span class="stat-val" id="task-domain">—</span></div>
      <hr class="sep">
      <div class="task-meta" id="task-instructions">Click Reset Episode to start.</div>
      <hr class="sep">
      <strong style="font-size:12px;color:var(--muted)">Expected Schema</strong>
      <table class="hint-table" id="schema-table"><tr><td colspan="2" style="color:var(--muted)">—</td></tr></table>
    </div>
  </div>

  <!-- Right col: Episode Stats -->
  <div class="card">
    <div class="card-header"><h2>Episode Stats</h2></div>
    <div class="card-body">
      <div class="stat-row"><span>Step</span><span class="stat-val" id="s-step">0 / 3</span></div>
      <div class="stat-row"><span>Cumulative Reward</span><span class="stat-val" id="s-reward">0.00 / 3.00</span></div>
      <div class="reward-bar-wrap"><div class="reward-bar" id="reward-bar" style="width:0%"></div></div>
      <div class="stat-row"><span>Last Step Reward</span><span class="stat-val" id="s-last">—</span></div>
      <div class="stat-row"><span>Episode Seed</span><span class="stat-val" id="s-seed">—</span></div>
      <div class="stat-row"><span>Dynamic Generation</span><span class="stat-val" id="s-dyn">—</span></div>
      <hr class="sep">
      <div class="btn-row">
        <button class="btn btn-primary" onclick="doReset()">Reset Episode</button>
      </div>
    </div>
  </div>

  <!-- Dirty Data -->
  <div class="card">
    <div class="card-header"><h2>Dirty Dataset</h2><span id="dirty-count" style="font-size:11px;color:var(--muted)"></span></div>
    <div class="card-body">
      <pre id="dirty-rows">Reset to load data...</pre>
    </div>
  </div>

  <!-- Submit cleaned -->
  <div class="card">
    <div class="card-header"><h2>Your Cleaned Dataset (JSON)</h2></div>
    <div class="card-body">
      <textarea id="cleaned-input" placeholder='{"cleaned_rows": [...], "operations_log": ["describe what you did"], "dropped_indices": []}'></textarea>
      <div class="btn-row">
        <button class="btn btn-success" onclick="doStep()">Submit Step</button>
        <button class="btn" style="background:#21262d;color:var(--text)" onclick="copyDirty()">Copy Dirty as Template</button>
      </div>
    </div>
  </div>

  <!-- Grader Feedback -->
  <div class="card full-width">
    <div class="card-header"><h2>Grader Feedback & Log</h2></div>
    <div class="card-body">
      <div class="log" id="log">Waiting for actions...</div>
    </div>
  </div>

</main>

<script>
let state = { obs: null, cumReward: 0, step: 0, seed: null };

function log(msg, cls='info') {
  const el = document.getElementById('log');
  const ts = new Date().toLocaleTimeString();
  el.innerHTML += `<div class="${cls}">[${ts}] ${msg}</div>`;
  el.scrollTop = el.scrollHeight;
}

function updateStats(reward, info, obs) {
  state.step = info?.step_count ?? state.step;
  state.cumReward = info?.cumulative_reward ?? state.cumReward;
  state.seed = info?.episode_seed ?? state.seed;
  document.getElementById('s-step').textContent = `${state.step} / 3`;
  document.getElementById('s-reward').textContent = `${state.cumReward.toFixed(2)} / 3.00`;
  document.getElementById('reward-bar').style.width = `${(state.cumReward/3)*100}%`;
  document.getElementById('s-last').textContent = reward !== undefined ? reward.toFixed(2) : '—';
  document.getElementById('s-seed').textContent = state.seed ?? '—';
  document.getElementById('s-dyn').textContent = info?.dynamic ? 'Yes (LLM-generated)' : 'No (static fallback)';
}

function updateTask(obs, info) {
  if (!obs || obs.done) return;
  const diff = obs.difficulty;
  const pill = document.getElementById('diff-pill');
  pill.textContent = diff.charAt(0).toUpperCase() + diff.slice(1);
  pill.className = `pill pill-${diff}`;
  document.getElementById('task-name').textContent = obs.task_name;
  document.getElementById('task-id').textContent = obs.task_id;
  document.getElementById('task-domain').textContent = info?.domain ?? obs.task_id.split('_')[0];
  document.getElementById('task-instructions').innerHTML = obs.instructions.replace(/\n/g, '<br>');
  document.getElementById('dirty-rows').textContent = JSON.stringify(obs.dirty_rows, null, 2);
  document.getElementById('dirty-count').textContent = `${obs.dirty_rows.length} rows`;
  const hint = obs.schema_hint;
  const tbl = document.getElementById('schema-table');
  tbl.innerHTML = Object.entries(hint).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}

async function doReset() {
  log('Calling POST /reset ...', 'info');
  try {
    const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'}});
    const obs = await r.json();
    state.cumReward = 0; state.step = 0;
    updateTask(obs, {domain: obs.task_id.split('_')[0], dynamic: false});
    updateStats(undefined, {step_count:0, cumulative_reward:0, episode_seed: null, dynamic: false}, obs);
    document.getElementById('cleaned-input').value = '';
    log(`Episode started. Task: ${obs.task_name} [${obs.difficulty}]`, 'ok');
    if (obs.feedback) log(obs.feedback, 'info');
  } catch(e) { log('Reset failed: ' + e.message, 'err'); }
}

async function doStep() {
  const raw = document.getElementById('cleaned-input').value.trim();
  if (!raw) { log('Paste your cleaned JSON first.', 'err'); return; }
  let action;
  try { action = JSON.parse(raw); } catch(e) { log('Invalid JSON: ' + e.message, 'err'); return; }
  if (!action.operations_log) action.operations_log = [];
  if (!action.dropped_indices) action.dropped_indices = [];
  log('Submitting step...', 'info');
  try {
    const r = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(action)});
    const result = await r.json();
    const {observation, reward, done, info} = result;
    updateStats(reward, info, observation);
    log(`Reward: ${reward.toFixed(2)}`, reward >= 0.7 ? 'ok' : reward >= 0.4 ? 'info' : 'err');
    if (observation.feedback) {
      observation.feedback.split('\n').forEach(l => { if(l.trim()) log(l, l.startsWith('✅') ? 'ok' : l.startsWith('❌') ? 'err' : 'info'); });
    }
    if (!done) {
      updateTask(observation, info);
      log(`Next: ${observation.task_name} [${observation.difficulty}]`, 'info');
      document.getElementById('cleaned-input').value = '';
    } else {
      log('Episode complete!', 'ok');
      document.getElementById('dirty-rows').textContent = 'Episode complete. Click Reset to start a new one.';
      document.getElementById('diff-pill').textContent = 'Done';
      document.getElementById('diff-pill').className = 'pill badge-blue';
    }
  } catch(e) { log('Step failed: ' + e.message, 'err'); }
}

function copyDirty() {
  const dirty = document.getElementById('dirty-rows').textContent;
  let rows;
  try { rows = JSON.parse(dirty); } catch { return; }
  const template = JSON.stringify({cleaned_rows: rows, operations_log: ["describe your changes here"], dropped_indices: []}, null, 2);
  document.getElementById('cleaned-input').value = template;
  log('Dirty data copied to editor as template.', 'info');
}
</script>
</body>
</html>"""
