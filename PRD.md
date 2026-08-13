# PRD: Simulation-Driven Marketing Intelligence Engine

**Type:** Proof of Concept (local-first, open-source LLMs)
**Status:** v1 architecture

---

## 1. One-line summary

A system that generates marketing campaigns grounded in layered brand/market
memory, then tests them against a population of data-derived synthetic personas
*before* any real ad spend — so a brand learns which creative works, for which
segment, and why, in minutes instead of weeks.

---

## 2. Why this exists (problem & opportunity)

For a house of 30+ brands, running real market research and live ad-testing for
every brand, launch, and creative decision is too slow and too expensive to do
at portfolio scale. Insight arrives after the money is already spent.

The opportunity is a **shared, reusable intelligence layer**: derive personas
once from real audience data, and let every brand test unlimited campaign ideas
against those personas cheaply and instantly.

Unlike a one-off research study

(which serves one brand once), this asset compounds — every brand and every
campaign that runs through it makes the shared market intelligence richer.

---

## 3. System shape: two modules + one evaluation layer

The system is deliberately structured as **two independent modules** that meet
at a single **evaluation layer**. This separation is the core design decision:
personas and campaigns are built by different pipelines, for different reasons,
and only come together at the moment of testing.

```
  MODULE A: PERSONA GENERATION       MODULE B: CAMPAIGN GENERATION
  ───────────────────────────       ─────────────────────────────
  raw audience data                  brand brief
        │                                  │
        ▼                                  ▼
  Segmentation                       [CHECKPOINT 1] Intake — human
  (cluster into segments)            answers 5-6 planning questions
        │                                  │  (→ working brief)
        ▼                                  ▼
  Persona Simulation                 Retrieve memory:
  (segment → rich persona)             • Brand (private)
        │                              • Market + Trend (global)
        │                                  │
        │                                  ▼
        │                            Analytical framework (auto-selected:
        │                            SWOT / Perceptual Map / 5 Forces / BMC)
        │                                  │  → positioning brief
        │                                  ▼
        │                            7Ps Marketing Mix
        │                                  │  → creative constraints
        │                                  ▼
        │                            Generate N campaign variants
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              EVALUATION LAYER — Traction Agent
   (test every variant against every persona; grounded
    scoring; synthesize pattern → ranked recommendation)
                       │
                       ▼
        [CHECKPOINT 2] Approval — marketer signs
        off on the winner before real ad spend
```

### Module A — Persona Generation
Turns raw audience data into a reusable population of synthetic personas.
- **Segmentation:** cluster raw engagement data into statistically distinct segments.
- **Persona Simulation:** turn each segment into a rich, queryable persona profile (identity, voice, buying triggers, objections) grounded in that segment's real stats.
- Output: 6–8 personas per brand. Built once, reused across every campaign test.

### Module B — Campaign Generation
Turns a brand brief into campaign variants — but as a **memory-grounded,
human-checkpointed workflow**, not a single prompt. This is the key correction
over a naive "brief in, copy out" call. Steps:

0. **Checkpoint 1 — Intake (human-in-the-loop):** before anything is generated,
   a planning step asks the user a small, fixed set of 5–6 questions (target
   segment, price posture, competitors to position against, the one core
   message, hard constraints, primary goal). Answers become the *working
   brief*. This is the step where a **more capable model** belongs (planning
   reasoning), while the high-volume simulation loop downstream runs on a cheap
   local model. For a reproducible demo the questions are displayed live and
   answers are read from a pre-filled file; in production the same function
   takes answers from the UI. See §5b.
1. **Retrieve brand memory (private):** brand identity, tone, stated competitors/customers.
2. **Retrieve market + trend memory (global):** abstracted, cross-brand patterns on what's resonating in this category right now. Never raw per-brand data — see §5.
3. **Apply an analytical framework → positioning brief:** auto-select one of
   SWOT / Perceptual Mapping / Porter's Five Forces / Business Model Canvas
   (based on the primary goal), consume brand + market memory, and output a
   positioning brief (where the brand sits, the whitespace, what to exploit or
   defend). Overridable; can also run several and synthesize. See §5a.
4. **Apply the 7Ps Marketing Mix → creative constraints:** always run at the
   execution step, translating the positioning brief into concrete constraints
   (price posture, proof elements, promotional angle, hard rules) so generated
   creative is consistent with the real marketing mix — e.g. a premium product
   never gets discount-led copy.
5. **Generate variants:** only now does the LLM write copy, conditioned on the
   working brief + brand/market/trend memory + positioning brief + 7Ps
   constraints.
- Output: N campaign variants, each with a distinct messaging angle.

### Evaluation Layer — Traction Agent
The single meeting point of the two modules.
- Tests each Module B variant against each Module A persona (grounded scoring — see §6).
- Synthesizes across all persona reactions per variant: traction tier, what's driving the pattern, targeting implication, and a concrete recommendation (scale / revise / narrow / kill).
- Surfaces results to a human for final approval.

---

## 4. Memory architecture (mocked for POC, architecturally real)

Four memory layers, each mapped to the store type that fits its access pattern.
For the POC all four are seeded with mock data in local files, but the
*interfaces* are real so they can be swapped for production stores later.

| Layer | Contains | Memory type | POC store | Prod store |
|---|---|---|---|---|
| **Brand (private)** | Per-brand identity, tone, competitors, customers, history | Episodic + semantic | JSON per brand | Per-tenant DB + vector namespace |
| **Global / Market** | Cross-brand, *anonymized* patterns ("in this category, X angle tends to outperform Y") | Semantic | JSON + simple vector sim | Shared vector DB |
| **Trend** | Currently-resonating cultural/creative signals per category | Semantic (time-decayed) | JSON | Vector DB + freshness TTL |
| **Framework (procedural)** | Business frameworks applied as method — SWOT, Perceptual Mapping, Porter's Five Forces, Business Model Canvas, 7Ps (see §5) | Procedural | JSON templates w/ schemas | Document store |

Retrieval at campaign-generation time pulls from all four and composes them into
the generation prompt.

---

## 5. Trust & privacy constraint (why global memory is safe)

Because this is a consulting/portfolio setup, one brand's data must never leak to
another. The rule: **global memory only ever contains abstracted, multi-brand
patterns — never attributable raw data.** Enforced structurally, not by asking an
LLM to be careful:
- Private memory is hard-partitioned per brand (POC: separate files; prod: per-tenant namespaces).
- A separate aggregation step extracts patterns from private memory, strips identifiers, and only promotes a pattern to global memory once it recurs across multiple brands.
- The generation step reads global memory but global memory has no brand identifiers to leak — the link was severed at write time, not read time.

(Full aggregation mechanism is documented for the pitch; POC ships mock global
memory that already looks post-abstraction.)

---

## 6. Business frameworks (procedural memory, applied as method)

The frameworks are not name-drops in a prompt — each is stored with a purpose,
a selection heuristic, the memory inputs it consumes, and an output schema it
must fill. They split into two roles by *when* they run:

**Analytical (strategy in) — ONE selected per run, auto-selected by goal:**

| Framework | Does what | Auto-selected when goal involves… |
|---|---|---|
| **SWOT** | Internal strengths/weaknesses × external opportunities/threats | (default / general strategic grounding) |
| **Perceptual Mapping** | Plots brand vs competitors on a 2D grid (e.g. price × quality) to find whitespace | new brand, launch, crowded category, differentiation |
| **Porter's Five Forces** | Structural industry competition (buyer/supplier power, substitutes, entrants, rivalry) | margin pressure, price war, defensibility |
| **Business Model Canvas** | Operational structure, partnerships, revenue/cost vs competitors | operational/cost advantage, sourcing, subscription |

Output of this step = a **positioning brief**.

**Executional (execution out) — ALWAYS applied:**

| Framework | Does what |
|---|---|
| **7Ps Marketing Mix** | Product, Price, Place, Promotion, People, Physical Evidence, Process → concrete **creative constraints** the generated variants must obey |

Flow: brand+market memory → analytical framework → positioning brief → 7Ps →
creative constraints → generation.

## 7. Human-in-the-loop checkpoints

The system has **two** human touchpoints — direction in, approval out:

1. **Checkpoint 1 — Intake (before generation):** user answers 5–6 planning
   questions that set strategic direction. Machine then does all the heavy
   lifting. (POC: fixed question set for a clean demo flow; Phase 2: a stronger
   planning model that asks only for real gaps in memory.)
2. **Checkpoint 2 — Approval (after evaluation):** the Traction Agent ranks
   variants; a marketer signs off on the winner before any real ad spend.

This matches the challenge brief's explicit ask for human-in-the-loop
checkpoints, and mirrors how a real consulting engagement runs: human sets
direction, machine executes, human approves output.

## 8. Grounded scoring (not free-floating LLM guesses)

Persona reactions are anchored to a **historical baseline** (synthetic for POC:
baseline response by segment × messaging angle). The LLM does not invent a score;
it *adjusts* the baseline by a bounded amount based on the specific copy's
execution for that persona. Final score = baseline + adjustment, clipped. Every
score stays traceable to "segment history" vs "copy-specific judgment."

---

## 9. Tech stack (local-first, open source)

| Concern | Choice | Notes |
|---|---|---|
| LLM inference | **Ollama or LM Studio**, local | Both expose an OpenAI-compatible `/v1/chat/completions` endpoint |
| LLM client | One thin wrapper pointed at `http://localhost:11434/v1` (Ollama) or LM Studio's port | Swappable via config/env; `api_key` is required-but-ignored. **Role-based routing:** the planning/intake step can point at a stronger model while the high-volume simulation loop uses a cheap fast local model — match model capability to task |
| Suggested models | `llama3.1:8b` / `mistral:7b` (fast) — any local chat model works | Low temperature (0.1–0.3) for parseable JSON output |
| Robust parsing | Custom JSON-extraction with retry | Small local models emit messier JSON than hosted APIs — must handle gracefully |
| Segmentation/clustering | scikit-learn (KMeans) + TF-IDF | Deterministic, offline, no GPU needed |
| Memory stores (POC) | Local JSON + a lightweight vector similarity | Interfaces real, backends swappable |
| Dashboard | Streamlit | For the video walkthrough |

**Design principle:** the architecture must be seamless and swappable — model
quality is explicitly secondary to a clean, robust, well-separated pipeline.
Any LLM (local or hosted) plugs into the same interface.

---

## 10. Repo structure

```
persona-sim-engine/
├── PRD.md
├── README.md
├── requirements.txt
├── config.py                  # model + endpoint + memory-path config
├── llm/
│   └── client.py              # OpenAI-compatible client (Ollama/LM Studio), JSON-safe
├── memory/
│   ├── brand_memory.py        # private, per-brand
│   ├── market_memory.py       # global, anonymized
│   ├── trend_memory.py        # trend signals
│   ├── framework_memory.py    # procedural frameworks
│   └── seed/                  # mock seed data for all four layers
├── module_a_personas/
│   ├── segmentation.py        # raw data → clusters
│   └── persona_simulation.py  # clusters → rich persona profiles
├── module_b_campaigns/
│   ├── intake.py              # Checkpoint 1: 5-6 planning questions → working brief
│   ├── frameworks_apply.py    # apply analytical framework → positioning brief; then 7Ps → constraints
│   └── campaign_generation.py # memory-grounded workflow → variants
├── evaluation/
│   ├── persona_reaction.py    # grounded per-(persona,variant) scoring
│   └── traction_agent.py      # synthesis across reactions → recommendation
├── data/
│   ├── generate_synthetic_data.py       # mock Indian D2C audience
│   └── generate_historical_campaigns.py # mock baseline table
├── dashboard/
│   └── app.py
└── run_pipeline.py            # orchestrates A + B → evaluation, end to end
```

---

## 11. Out of scope for POC (→ pitch/roadmap as Phase 2)

- Real Meta/Instagram API ingestion (POC uses synthetic Indian D2C audience data)
- Live production multi-tenant infra, auth, access control enforcement
- The full automated aggregation/abstraction pipeline (POC ships mock post-abstraction global memory)
- Fine-tuning (all agents run on prompted local base models)

---

## 12. Success criteria

- [ ] Runs fully locally on an open-source LLM via Ollama/LM Studio, one config change to switch model
- [ ] Role-based model routing works (stronger model for intake/planning, cheap model for simulation loop)
- [ ] `--mock` mode runs the whole pipeline with zero LLM calls (fast structural validation)
- [ ] Module A produces 6–8 data-derived personas
- [ ] Checkpoint 1 intake displays 5–6 questions and folds answers into the working brief
- [ ] Module B visibly retrieves from all four memory layers, applies an analytical framework → positioning brief, then 7Ps → constraints, before generating (not a bare prompt)
- [ ] Evaluation layer scores every variant × persona with traceable baseline+adjustment
- [ ] Traction Agent outputs a ranked, actionable recommendation per variant
- [ ] Checkpoint 2 approval step present in the dashboard
- [ ] Clean module separation — A and B runnable independently, meeting only at evaluation
