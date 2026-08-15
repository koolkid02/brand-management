# Simulation-Driven Marketing Intelligence Engine

A POC that generates marketing campaigns grounded in layered brand/market memory,
then tests them against data-derived synthetic personas *before* any real ad
spend. Runs fully locally on open-source LLMs via Ollama.

Full architecture, design rationale, and success criteria: **[PRD.md](PRD.md)**.
This README is the practical "how do I run it" companion.

## What's here

- **Module A** — turns raw audience data into a reusable population of synthetic
  personas (segmentation + LLM-authored narrative).
- **Module B** — a bounded agentic loop that turns a brand brief into a small
  set of genuinely refined campaign variants: adaptive intake → strategy
  framework + 7Ps → ideate/critique/refine.
- **Module C** — the Traction Agent: tests every variant against every persona
  with grounded (baseline + bounded LLM adjustment) scoring, ranks them, and
  writes the human-approved winner back into brand memory.
- **Dashboard** — the primary way to interact with all of the above.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com), running locally, with these three models pulled:

  ```bash
  ollama pull llama3.2:latest      # "simulation" role -- cheap, high-volume
  ollama pull gemma4:e2b           # "planning" role -- stronger, once-per-run reasoning
  ollama pull nomic-embed-text     # "embedding" role -- vector similarity
  ```

  Model choice is swappable via env vars (see [Configuration](#configuration)) —
  any local chat model works, these are just the defaults.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every command below is run from the repo root, as a module (`python -m ...`),
so `config.py`/`llm/`/etc. resolve regardless of your shell's working directory.

## Quick start: the dashboard

```bash
streamlit run dashboard/app.py
```

Open the printed local URL. In the sidebar: pick a brand, toggle **Mock / demo
mode** (on = instant, zero LLM calls, template-based output for a structural
walkthrough; off = real local LLM calls, several minutes end-to-end), and
**Start Intake**. The "Personas" tab next to Campaign Setup lets you browse
Module A's synthetic audience directly.

The dashboard drives all three checkpoints: adaptive intake with follow-ups,
strategy review (approve or redirect, capped at 2 loop-backs), and evaluation
+ approval — the only action anywhere in the app that writes an outcome back
into brand memory.

## Running the pipeline by hand

Every script below supports `--mock` (zero LLM calls, deterministic templates
— useful for a fast structural check) and prints its own progress/method
provenance.

**Module A — build the persona population (run once, shared across brands):**

```bash
python data/generate_synthetic_data.py --n-customers 2000 --seed 42
python -m module_a_personas.segmentation
python -m module_a_personas.persona_simulation --mock
```

**Module B — generate campaign variants for one brand:**

```bash
python -m module_b_campaigns.intake --brand-id vamp_streetwear --mock
python -m module_b_campaigns.frameworks_apply --mock
python -m module_b_campaigns.ideation --mock
```

`intake.py` also supports `--source inline` and `--followup-mode
{simulated,interactive}` for scripted or live follow-up answering outside the
dashboard.

**Data — synthetic historical baselines (segment × messaging-angle scores,
used by Module C's grounded scoring):**

```bash
python -m data.generate_historical_campaigns --seed 42
```

**Module C — evaluate and rank a brand's variants:**

```bash
python -m evaluation.persona_reaction --mock       # one variant, all personas (smoke test)
python -m evaluation.traction_agent --brand-id vamp_streetwear --mock
```

**Memory — retrieval and the Checkpoint-2 write-back:**

```bash
python -m memory.market_memory --query "size and fit reassurance" --mock
python -m memory.trend_memory --query "festive capsule drop" --category fashion_d2c --mock
python -m memory.outcome_memory --brand-id vamp_streetwear --variant-id v1 \
    --evaluation-result path/to/eval.json --mock
```

`memory/market_memory.py` with no `--query` re-runs cross-brand pattern
aggregation instead of retrieving.

## Configuration

`config.py` defines three model roles, each overridable via env var:

| Role | Purpose | Default model | Env override |
|---|---|---|---|
| `simulation` | Cheap, high-volume (scales with N) | `llama3.2:latest` | `SIMULATION_MODEL`, `SIMULATION_BASE_URL`, `SIMULATION_TEMPERATURE` |
| `planning` | Stronger, once-per-run reasoning | `gemma4:e2b` | `PLANNING_MODEL`, `PLANNING_BASE_URL`, `PLANNING_TEMPERATURE` |
| `embedding` | Vector similarity | `nomic-embed-text:latest` | `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL` |

All three point at `http://localhost:11434/v1` (Ollama's default) unless
overridden — any OpenAI-compatible endpoint (Ollama, LM Studio, a hosted API)
works.

## Repo structure

```
config.py                  # model + endpoint config (role-based routing)
llm/client.py               # OpenAI-compatible client, JSON-safe with retry
memory/
  brand_memory.py           # private, per-brand (read + write)
  market_memory.py          # global, anonymized cross-brand patterns
  trend_memory.py           # time-decayed trend signals
  framework_memory.py       # procedural business frameworks
  outcome_memory.py         # Checkpoint-2 write-back into brand/market memory
  seed/                     # seed data for all memory layers
module_a_personas/
  segmentation.py           # raw audience data -> clusters
  persona_simulation.py     # clusters -> rich persona profiles
module_b_campaigns/
  intake.py                 # Checkpoint 1: adaptive intake -> working brief
  frameworks_apply.py       # analytical framework -> positioning; 7Ps -> constraints
  ideation.py                # ideate (wide) -> critique -> refine -> variants
  agent_loop.py              # bounded agentic orchestrator tying it all together
evaluation/
  persona_reaction.py       # grounded per-(persona, variant) scoring
  traction_agent.py         # per-variant synthesis + cross-variant ranking
data/
  generate_synthetic_data.py           # mock Indian D2C audience
  generate_historical_campaigns.py     # synthetic segment x angle baselines
dashboard/
  app.py                    # Streamlit UI -- the only file that imports streamlit
```

## Notes

- `--mock` mode is zero LLM calls everywhere, by design (a fast structural
  check and a live-demo fallback) — every generation step has a deterministic
  template that doubles as the fallback when a real LLM call fails after
  retries. Generated output always carries a `generation_method` field
  (`"llm" | "mock" | "llm_fallback"`) so provenance is traceable, and the
  dashboard surfaces a warning banner whenever something silently fell back.
- Data flows through `data/processed/*.json` between steps when run by hand;
  the dashboard holds the same state in-memory via `agent_loop.AgentState`.
