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
  MODULE A: PERSONA GENERATION       MODULE B: CAMPAIGN GENERATION (bounded agentic loop)
  ───────────────────────────       ───────────────────────────────────────────────────
  raw audience data                  brand brief
        │                                  │
        ▼                                  ▼
  Segmentation                       [CHECKPOINT 1] Intake — adaptive:
  (cluster into segments)            baseline questions + up to 3 dynamic
        │                            follow-ups on gaps/ambiguity
        ▼                            (→ working brief) · via dashboard
  Persona Simulation                        │
  (segment → rich persona)                  ▼
        │                            Retrieve memory:
        │                              • Brand (private)
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
        │                       ┌──▶ [CHECKPOINT 1.5] Strategy Review — human
        │                       │    approves direction, or redirects
        │                       │        │                    │
        │                       │     approve            redirect (≤2 loops)
        │                       │        │                    │
        │                       └────────┼────────────────────┘
        │                                ▼
        │                          Ideate (wide): 8-10 raw creative
        │                          concepts across creative angles
        │                                  │
        │                                  ▼
        │                          Critique: score/cut against brief,
        │                          positioning, hard rules (reasons stated)
        │                                  │
        │                                  ▼
        │                          Refine: polish the surviving
        │                          strongest concepts
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              EVALUATION LAYER — Traction Agent
   (test every refined variant against every persona; grounded
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
Turns a brand brief into a small set of genuinely strong campaign variants — not a
single "brief in, copy out" call, and not a one-shot batch of first drafts either.
Module B runs as a **bounded agentic loop**: the phase sequence is fixed (intake →
strategy → ideation → approval), but the model makes real, LLM-judged decisions
within and between phases — how many follow-up questions to ask, whether to loop
back after a strategy review, which concepts survive critique — rather than
executing a rigid script. All human touchpoints happen through a live dashboard
(Streamlit), not a pre-filled answers file or terminal prompts; the file-based mode
is kept only as a scripted path for reproducible verification runs, not the primary
experience.

0. **Checkpoint 1 — Intake (adaptive):** the planning model asks the fixed baseline
   questions (target segment, price posture, competitors to position against, the
   one core message, hard constraints, primary goal) via the dashboard, then
   reviews the answers for gaps, vagueness, or interesting tensions and asks up to
   3 targeted follow-up questions before finalizing the *working brief* — real
   back-and-forth, not a form. (POC: bounded to ≤3 follow-ups on the same fixed
   baseline set, for a predictable demo; Phase 2: fully open-ended discovery.) This
   is where the **planning-role model** belongs (stronger, once-per-run reasoning);
   the high-volume ideation loop downstream still runs on the cheap simulation
   model.
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

   **[Checkpoint 1.5 — Strategy Review, human-in-the-loop]:** before any creative
   is generated, the dashboard shows the selected framework, the positioning
   brief, and the 7Ps constraints. The human approves (proceed to ideation) or
   redirects with feedback ("wrong framework," "don't position against that
   competitor") — a redirect re-runs steps 3-4 with that feedback folded in,
   capped at 2 loop-backs so the POC stays bounded and demoable.
5. **Ideate (wide):** generate 8-10 raw creative concepts spanning the angle pool
   (simulation role — the high-volume step).
6. **Critique:** one planning-role pass reviews all raw concepts against the
   brief, positioning brief, and hard rules — scores each for on-brief adherence,
   differentiation, and hard-rule compliance, and selects the strongest subset to
   carry forward, with a stated reason for every cut (transparent process, not a
   black box).
7. **Refine:** surviving concepts get a genuine polish pass (planning role)
   informed by the critique feedback — first drafts never ship as final variants.
- Output: a small set (e.g. 3-5) of refined campaign variants, each with a
  distinct messaging angle and a visible critique rationale, ready for the
  evaluation layer.

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

## 4a. Memory write-back (this is how "compounds" actually happens)

§2 promises this asset compounds — every brand and every campaign that runs
through it makes the shared market intelligence richer — but §4 above only
describes the *read* side. This section describes the write side: how a real
campaign outcome turns back into new memory.

**Trigger:** Checkpoint 2 approval, and only Checkpoint 2 approval. When a
marketer signs off on a winning variant after evaluation, that outcome is
recorded as two things in the *winning brand's own private memory*:
- An **episodic** entry in `episodic.history` — a factual record ("ran this
  campaign, on this date, this was the angle").
- A **semantic** entry in `insights` — the same generalized, tagged pattern
  shape every hand-seeded insight already uses, so it flows through the exact
  same aggregation path (§5) as the insights the brands launched with.

Only a human-approved outcome triggers this — not raw automated Traction
Agent scoring. This mirrors why §7 has a human checkpoint there at all: an
approval is a real signal about what actually mattered enough to ship, an
automated score alone is not.

The one structural difference from every other insight in this system: the
tag vocabulary is no longer frozen at authoring time for this path. A new
campaign outcome might not match any existing tag, so a new tag can be
minted at runtime — but only after checking whether an existing tag already
means the same thing (matched by meaning, via embedding similarity, not by
exact string), so the vocabulary grows without silently fragmenting into
near-duplicate tags. New tags are registered into the exact same vocabulary
`market_memory.py`'s aggregation already depends on — there is no second,
parallel tagging system.

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

Insights written back via Checkpoint 2 approval (§4a) go through this
identical aggregation path as hand-seeded insights — same tag-gated
grouping, same 2-brand recurrence rule before anything promotes to global
memory, same category scoping. The abstraction into a brand-agnostic
canonical pattern statement happens at write time, before the tag is ever
registered, preserving "the link was severed at write time, not read time"
even for machine-derived insights, not just hand-authored ones.

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

The system has **three** human touchpoints — direction in, strategic alignment
check, approval out:

1. **Checkpoint 1 — Intake (before generation, adaptive):** user answers the
   fixed baseline questions that set strategic direction, then answers up to 3
   dynamic follow-up questions the planning model asks on whatever's vague or
   under-specified. (POC: bounded follow-up count on a fixed baseline set, for a
   clean demo flow; Phase 2: a stronger planning model that asks only for real
   gaps in memory, fully open-ended.)
2. **Checkpoint 1.5 — Strategy Review (after positioning + 7Ps, before creative
   execution):** the human sees the selected framework, positioning brief, and
   creative constraints, and either approves or redirects with feedback — a
   redirect re-runs framework/7Ps application, capped at 2 loop-backs.
3. **Checkpoint 2 — Approval (after evaluation):** the Traction Agent ranks
   variants; a marketer signs off on the winner before any real ad spend.
   This approval is also what triggers the memory write-back described in
   §4a — an automated Traction Agent score alone never does.

This matches the challenge brief's explicit ask for human-in-the-loop
checkpoints, and mirrors how a real consulting engagement runs — now more
literally: human sets direction, strategic alignment is checked before any
creative work starts, machine executes, human approves output.

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
| LLM client | One thin wrapper pointed at `http://localhost:11434/v1` (Ollama) or LM Studio's port | Swappable via config/env; `api_key` is required-but-ignored. **Role-based routing:** the planning/intake step can point at a stronger model while the high-volume simulation loop uses a cheap fast local model — match model capability to task. In Module B specifically: intake follow-up generation, strategy-redirect handling, critique, and refinement all run on `"planning"`; wide/raw ideation (and Module A's persona simulation) run on `"simulation"` — the same once-per-run-vs-scales-with-N rule, now covering more steps |
| Suggested models | `llama3.1:8b` / `mistral:7b` (fast) — any local chat model works | Low temperature (0.1–0.3) for parseable JSON output |
| Robust parsing | Custom JSON-extraction with retry | Small local models emit messier JSON than hosted APIs — must handle gracefully |
| Segmentation/clustering | scikit-learn (KMeans) + TF-IDF | Deterministic, offline, no GPU needed |
| Memory stores (POC) | Local JSON + a lightweight vector similarity | Interfaces real, backends swappable |
| Dashboard | Streamlit | Core interaction surface for Module B — adaptive intake conversation, strategy review, variant approval. Not a post-hoc demo aid; the primary way a human interacts with Checkpoints 1, 1.5, and 2 |

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
│   ├── intake.py              # Checkpoint 1: adaptive baseline + follow-up questions → working brief
│   ├── frameworks_apply.py    # analytical framework → positioning brief; 7Ps → constraints; supports redirect/re-apply
│   ├── ideation.py            # ideate (wide) → critique → refine → final variants
│   └── agent_loop.py          # bounded agentic orchestrator: phases + checkpoints + loop-back/follow-up decisions
├── evaluation/
│   ├── persona_reaction.py    # grounded per-(persona,variant) scoring
│   └── traction_agent.py      # synthesis across reactions → recommendation
├── data/
│   ├── generate_synthetic_data.py       # mock Indian D2C audience
│   └── generate_historical_campaigns.py # mock baseline table
├── dashboard/
│   └── app.py                 # Streamlit UI: intake conversation, strategy review, variant approval
└── run_pipeline.py            # orchestrates A + B → evaluation, end to end
```
`module_b_campaigns/` supersedes the earlier flat `campaign_generation.py` script —
its responsibilities are now split between `ideation.py` (the ideate/critique/refine
logic) and `agent_loop.py` (the orchestrator that ties phases and checkpoints together).

---

## 11. Out of scope for POC (→ pitch/roadmap as Phase 2)

- Real Meta/Instagram API ingestion (POC uses synthetic Indian D2C audience data)
- Live production multi-tenant infra, auth, access control enforcement
- The full automated aggregation/abstraction pipeline (POC ships mock post-abstraction global memory — note the §4a write-back's tag-matching is still a closed-vocabulary lookup extended by embedding similarity at write time, not free-text NLP clustering: new tags still require explicit registration and promotion still requires 2+ brand recurrence, so this stays out of scope, not partially built)
- Fine-tuning (all agents run on prompted local base models)
- Memory grounded in external case studies / marketing research papers, in addition to the self-generated campaign-outcome insights §4a produces
- A budget estimation tool integrated into the campaign planning agent, with cost modeling that varies by campaign type (offline media costs have a different structure than online/social costs)
- A shelved-campaign memory: lessons learned from rejected, not just approved, campaigns — a natural future extension of the §4a write-back mechanism, triggered by rejection instead of Checkpoint 2 approval

---

## 12. Success criteria

- [ ] Runs fully locally on an open-source LLM via Ollama/LM Studio, one config change to switch model
- [ ] Role-based model routing works (stronger model for intake/planning, cheap model for simulation loop)
- [ ] `--mock` mode runs the whole pipeline with zero LLM calls (fast structural validation)
- [ ] Module A produces 6–8 data-derived personas
- [ ] Checkpoint 1 intake asks the fixed baseline questions via the dashboard, then asks up to 3 dynamic follow-ups on gaps/ambiguity before finalizing the working brief
- [ ] Module B visibly retrieves from all four memory layers, applies an analytical framework → positioning brief, then 7Ps → constraints, before generating (not a bare prompt)
- [ ] Checkpoint 1.5 strategy review shows the selected framework/positioning brief/constraints in the dashboard and supports a bounded (≤2) redirect loop-back
- [ ] Module B's ideation step generates a wider raw concept set, critiques/cuts it with stated reasons, and only refines the survivors — never ships a first draft as a final variant
- [ ] Evaluation layer scores every variant × persona with traceable baseline+adjustment
- [ ] Traction Agent outputs a ranked, actionable recommendation per variant
- [ ] Checkpoint 2 approval step present in the dashboard
- [ ] Clean module separation — A and B runnable independently, meeting only at evaluation
- [ ] Checkpoint 2 approval writes the outcome back as a new tagged insight + history entry in the winning brand's memory, and a newly-recurring pattern promotes to market memory on the next aggregation run
